using CSV, DataFrames
using LinearAlgebra
using Random

# ---------------------------
# 1. Load Nodes and Edges
# ---------------------------

nodes_file = "/Users/vaw1/Downloads/OGB/node_regions_clean.csv"
edges_file = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv" 


nodes_df = CSV.read(nodes_file, DataFrame; delim=';')
edges_df = CSV.read(edges_file, DataFrame; delim=';')

# ---------------------------
# Define regions and fractions
# ---------------------------

# Example: 6 regions with 2-5% active nodes each
region_fractions = Dict(
    "fiber tracts" => 0.05,
    "bgr" => 0.02,
    "CUL4" => 0.03,
    "CBXmo" => 0.04,
    "VIS" => 0.02,
    "RSP" => 0.05
)

# ---------------------------
# Define Node and Edge structs
# ---------------------------

mutable struct Node
    id::Int
    pos::Vector{Float64}
    isBorder::Bool
    regions::Vector{String}
    incoming::Vector{Int}   # IDs of incoming edges
    outgoing::Vector{Int}   # IDs of outgoing edges
    entropy::Vector{Float64}  # entropy contributions per outcome
end

mutable struct Edge
    id::Int
    node1::Int
    node2::Int
end

# ---------------------------
# Build network dictionaries
# ---------------------------

node_dict = Dict{Int, Node}()
edge_dict = Dict{Int, Edge}()

# Create nodes
for row in eachrow(nodes_df)
    regions_clean = replace(replace(row[:regions], "["=>""), "]"=>"")
    regions_clean = replace(regions_clean, "'" => "")
    regions_clean = [strip(regions_clean)]  # single-element array
    node_dict[row[:id]] = Node(
        row[:id],
        [row[:pos_x], row[:pos_y], row[:pos_z]],
        row[:isAtSampleBorder] == 1,
        regions_clean,
        Int[], Int[],
        Float64[]  # placeholder for entropy
    )
end

# Create edges and populate adjacency
for row in eachrow(edges_df)
    edge_dict[row[:id]] = Edge(row[:id], row[:node1id], row[:node2id])
    push!(node_dict[row[:node1id]].outgoing, row[:id])
    push!(node_dict[row[:node2id]].incoming, row[:id])
end

# ---------------------------
# Entropy flow struct
# ---------------------------
mutable struct EntropyFlow
    name::Symbol
    values::Vector{Float64}  # entropy per outcome
end

# ---------------------------
# BV delta operator for entropy
# ---------------------------
function delta_entropy(flow::EntropyFlow; alpha=1.0, shift=0.0)
    # Deform extreme entropy values
    deformed = [v^alpha for v in flow.values]
    # Shift to smooth singularities
    shifted = [v + shift for v in deformed]
    # Normalize to conserve total entropy
    total = sum(shifted)
    normalized = shifted .* (sum(flow.values)/total)
    return EntropyFlow(Symbol(flow.name, "_resolved"), normalized)
end

# ---------------------------
# Gerstenhaber-style bracket
# ---------------------------
function bracket_entropy(a::EntropyFlow, b::EntropyFlow; alpha=1.0, shift=0.0)
    combined_values = a.values .+ b.values  # additive combination
    combined = EntropyFlow(Symbol(a.name, "_", b.name), combined_values)
    return delta_entropy(combined; alpha=alpha, shift=shift)
end


# ---------------------------
# 1. BV delta operator
# ---------------------------
# This represents the Batalin-Vilkovisky differential acting on entropy flows.
# It deforms, shifts, and normalizes the entropy vector while preserving total entropy.
function bv_delta_operator(flow::EntropyFlow; alpha=1.0, shift=0.0)
    # 1. Deformation: element-wise exponentiation
    deformed = [v^alpha for v in flow.values]
    
    # 2. Shift: additive shift per element
    shifted = [v + shift for v in deformed]
    
    # 3. Normalization: ensure total entropy remains the same
    total = sum(shifted)
    normalized = shifted .* (sum(flow.values)/total)
    
    return EntropyFlow(Symbol(flow.name, "_bv"), normalized)
end

# ---------------------------
# 2. Gerstenhaber bracket
# ---------------------------
# This is the Gerstenhaber bracket between two entropy flows.
# Conceptually, it combines derivations (entropy contributions) from two sources.
function gerstenhaber_bracket(a::EntropyFlow, b::EntropyFlow; alpha=1.0, shift=0.0)
    # 1. Combine flows (discrete analog of the bracket)
    combined_values = a.values .+ b.values
    combined = EntropyFlow(Symbol(a.name, "_bracket_", b.name), combined_values)
    
    # 2. Apply BV delta operator to the combined flow
    return bv_delta_operator(combined; alpha=alpha, shift=shift)
end


# 1. Compute initial total entropy
E_total = sum([sum(node.entropy) for node in values(node_dict)])


# ---------------------------
# Select active nodes per region
# ---------------------------

function select_active_nodes(node_dict::Dict{Int, Node}, region_fractions::Dict{String, Float64})
    active_ids = Set{Int}()
    for (region, fraction) in region_fractions
        # Filter nodes in region
        region_nodes = [node for node in values(node_dict) if region in node.regions]
        num_selected = max(1, round(Int, fraction * length(region_nodes)))
        # Random selection using randperm
        indices = randperm(length(region_nodes))[1:num_selected]
        selected_nodes = region_nodes[indices]
        # Add to global set
        union!(active_ids, [node.id for node in selected_nodes])
    end
    return active_ids
end


#-----------------------------
# Multiregion Entropy Flow
#-----------------------------
# Conserves per region entropy
function propagate_entropy_multi_regions_conserve_regional_entropy(node_dict::Dict{Int, Node}, edge_dict::Dict{Int, Edge}, 
                                         num_outcomes::Int, active_ids::Set{Int}; iterations=5)

    # Initialize all nodes with max entropy
    for node in values(node_dict)
        node.entropy = fill(log(num_outcomes), num_outcomes)
    end

    # Iterative propagation
    for iter in 1:iterations
        for node in values(node_dict)
            if node.id ∈ active_ids
                incoming_flows = EntropyFlow[]
                for eid in node.incoming
                    e = edge_dict[eid]
                    src_node = node_dict[e.node1 == node.id ? e.node2 : e.node1]
                    push!(incoming_flows, EntropyFlow(Symbol("n$(src_node.id)"), src_node.entropy))
                end

                if !isempty(incoming_flows)
                    combined = incoming_flows[1]
                    for f in incoming_flows[2:end]
                        combined = bracket_entropy(combined, f; alpha=0.9, shift=0.1)
                    end
                    node.entropy = combined.values
                end
            end
        end
    end
end

# This will allow for regional entropy to change while keeping system entropy constant (closed system).
function propagate_entropy_multi_regions(node_dict::Dict{Int, Node}, edge_dict::Dict{Int, Edge}, 
                                         num_outcomes::Int, active_ids::Set{Int}; iterations=5)

    # Initialize nodes with max entropy
    for node in values(node_dict)
        node.entropy = fill(log(num_outcomes), num_outcomes)
    end

    for iter in 1:iterations
        for node in values(node_dict)
            if node.id ∈ active_ids
                incoming_flows = EntropyFlow[]
                
                # Collect entropy from incoming edges
                for eid in node.incoming
                    e = edge_dict[eid]
                    src_node = node_dict[e.node1 == node.id ? e.node2 : e.node1]
                    push!(incoming_flows, EntropyFlow(Symbol("n$(src_node.id)"), src_node.entropy))
                end

                # Combine incoming flows using Gerstenhaber bracket + BV delta
                if !isempty(incoming_flows)
                    combined = incoming_flows[1]
                    for f in incoming_flows[2:end]
                        combined = gerstenhaber_bracket(combined, f; alpha=0.9, shift=0.1)
                    end
                    node.entropy = combined.values
                end
            end
        end
    end
end


# ---------------------------
# Propagate entropy through selected nodes
# ---------------------------
function propagate_entropy_selected(node_dict::Dict{Int, Node}, edge_dict::Dict{Int, Edge}, 
                                    num_outcomes::Int, region::String, fraction::Float64=0.1; iterations=5)

    # 1. Filter nodes in region
    region_nodes = [node for node in values(node_dict) if region in node.regions]

    # 2. Randomly select fraction of nodes
    num_selected = max(1, round(Int, fraction * length(region_nodes)))
    selected_nodes = Random.sample(region_nodes, num_selected; replace=false)
    selected_ids = Set(node.id for node in selected_nodes)

    # 3. Initialize all nodes with max entropy
    for node in values(node_dict)
        node.entropy = fill(log(num_outcomes), num_outcomes)
    end

    # 4. Propagation loop
    for iter in 1:iterations
        for node in values(node_dict)
            # Only propagate for selected nodes
            if node.id ∈ selected_ids
                incoming_flows = EntropyFlow[]
                for eid in node.incoming
                    e = edge_dict[eid]
                    src_node = node_dict[e.node1 == node.id ? e.node2 : e.node1]
                    push!(incoming_flows, EntropyFlow(Symbol("n$(src_node.id)"), src_node.entropy))
                end

                if !isempty(incoming_flows)
                    combined = incoming_flows[1]
                    for f in incoming_flows[2:end]
                        combined = bracket_entropy(combined, f; alpha=0.9, shift=0.1)
                    end
                    node.entropy = combined.values
                end
            end
        end
    end

    return selected_ids
end

# ---------------------------
# 7. Propagate entropy through network
# ---------------------------
function propagate_entropy(node_dict::Dict{Int, Node}, edge_dict::Dict{Int, Edge}, num_outcomes::Int; iterations=5)
    # Initialize all nodes with max entropy
    for node in values(node_dict)
        node.entropy = fill(log(num_outcomes), num_outcomes)  # uniform max entropy
    end

    for iter in 1:iterations
        for node in values(node_dict)
            incoming_flows = EntropyFlow[]
            for eid in node.incoming
                e = edge_dict[eid]
                src_node = node_dict[e.node1 == node.id ? e.node2 : e.node1]
                push!(incoming_flows, EntropyFlow(Symbol("n$(src_node.id)"), src_node.entropy))
            end

            if !isempty(incoming_flows)
                combined = incoming_flows[1]
                for f in incoming_flows[2:end]
                    combined = bracket_entropy(combined, f; alpha=0.9, shift=0.1)
                end
                node.entropy = combined.values
            end
        end
    end
end

function region_totals(node_dict::Dict{Int, Node})
    totals = Dict{String, Float64}()
    for node in values(node_dict)
        region = node.regions[1]
        totals[region] = get(totals, region, 0.0) + sum(node.entropy)
    end
    return totals
end

function conserve_total_entropy(node_dict::Dict{Int, Node}, E_total::Float64)
    totals = region_totals(node_dict)
    total_current = sum(values(totals))
    scaling = Dict{String, Float64}()
    for (region, total) in totals
        scaling[region] = total != 0 ? (E_total * (total / total_current)) / total : 1.0
    end
    for node in values(node_dict)
        region = node.regions[1]
        node.entropy .= node.entropy .* scaling[region]
    end
end


# ---------------------------
# 8. Trace normalized contributions for an outcome
# ---------------------------
function normalized_entropy_contributions(flows::Vector{Vector{Float64}}, outcome_index::Int)
    contributions = [f[outcome_index] for f in flows]
    total = sum(contributions)
    return [c/total for c in contributions]
end

# ---------------------------
# 9. Run simulation
# ---------------------------
num_outcomes = 3  # example: 3 discrete outcomes
active_node_ids = select_active_nodes(node_dict, region_fractions)
propagate_entropy_multi_regions(node_dict, edge_dict, num_outcomes, active_node_ids)
conserve_total_entropy(node_dict, E_total)

#----------------------------
# Summarize entropy by region
#----------------------------
function nodes_by_region(node_dict::Dict{Int, Node})
    region_nodes = Dict{String, Vector{Node}}()
    for node in values(node_dict)
        region = node.regions[1]  # single-region assumption
        if !haskey(region_nodes, region)
            region_nodes[region] = Node[]
        end
        push!(region_nodes[region], node)
    end
    return region_nodes
end

function summarize_entropy_per_region(node_dict::Dict{Int, Node})
    region_nodes = nodes_by_region(node_dict)
    region_summary = Dict{String, Vector{Float64}}()

    for (region, nodes) in region_nodes
        if !isempty(nodes)
            # Sum entropy vectors element-wise
            total_entropy = zeros(length(nodes[1].entropy))
            for node in nodes
                total_entropy .+= node.entropy
            end
            # Optionally normalize per outcome
            total_entropy ./= length(nodes)  # average per node
            region_summary[region] = total_entropy
        end
    end
    return region_summary
end




region_summary = summarize_entropy_per_region(node_dict)

println("Entropy summary per region:")
for (region, entropy_vec) in region_summary
    println("Region: $region, entropy vector: $entropy_vec, sum=", sum(entropy_vec))
end

# ---------------------------
# 5. Print results
# ---------------------------
println("Active nodes across all regions: ", active_node_ids)
for node_id in active_node_ids
    node = node_dict[node_id]
    println("Node $(node.id) (region=$(node.regions[1])) entropy: ", node.entropy, " sum=", sum(node.entropy))
end
