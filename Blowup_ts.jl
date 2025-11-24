###############################################################
#                   LIBRARIES
###############################################################

using CSV, DataFrames, Random, Statistics
using Plots
using StatsBase

NODES="/Users/vaw1/Downloads/OGB/node_regions_clean.csv"
EDGES="/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"


###############################################################
#                   DATA STRUCTURES
###############################################################

mutable struct Node
    id::Int
    regions::Vector{String}
    incoming::Vector{Int}
    outgoing::Vector{Int}
    entropy_fwd::Vector{Float64}
    entropy_bwd::Vector{Float64}
    entropy_overlap::Vector{Float64}
end

struct Edge
    id::Int
    node1::Int
    node2::Int
end

###############################################################
#                   LOAD NODES / EDGES
###############################################################

println("Loading nodes...")
node_df = CSV.File(NODES; delim=';') |> DataFrame
node_dict = Dict{Int, Node}()

function parse_regions(r)
    # convert "['bgr']" → ["bgr"]
    s = String(r)
    s = replace(s, '[' => "", ']' => "", "'" => "")
    parts = split(s, ",")
    return strip.(parts)
end

for row in eachrow(node_df)
    regs = parse_regions(row.regions)

    node_dict[row.id] = Node(
        row.id,
        regs,
        Int[], Int[],
        Float64[], Float64[], Float64[]
    )
end

println("Loading edges...")
edge_df = CSV.File(EDGES; delim=';') |> DataFrame
edge_dict = Dict{Int, Edge}()

for row in eachrow(edge_df)
    edge_dict[row.id] = Edge(row.id, row.node1id, row.node2id)
    push!(node_dict[row.node1id].outgoing, row.id)
    push!(node_dict[row.node2id].incoming, row.id)
end

###############################################################
#            REGION FRACTIONS (change as needed)
###############################################################

# example – you can extend this to 6–7 regions
region_fractions = Dict(
    "bgr" => 0.02,
    "CUL4" => 0.03,
    "fiber tracts" => 0.03
)

###############################################################
#           ACTIVE NODE SELECTION (FORWARD FLOW)
###############################################################

function select_active_nodes(node_dict, region_fractions)
    active_ids = Set{Int}()
    for (region, fraction) in region_fractions
        region_nodes = [n.id for n in values(node_dict) if region in n.regions]
        if isempty(region_nodes)
            @warn "Region $region has no nodes"
            continue
        end
        k = max(1, round(Int, fraction * length(region_nodes)))
        k = min(k, length(region_nodes))
        # randomly choose k nodes WITHOUT needing sample()
        perm = randperm(length(region_nodes))
        chosen = region_nodes[perm[1:k]]

        # add the chosen IDs to final set properly
        union!(active_ids, chosen)
    end
    return active_ids
end

###############################################################
#        ACTIVE NODE SELECTION (BACKWARD FLOW)
###############################################################

function select_backward_nodes(node_dict, region_fractions, active_forward::Set{Int}, overlap_fraction=0.3)
    active_backward = Set{Int}()

    for (region, frac) in region_fractions
        region_nodes = [n.id for n in values(node_dict) if region in n.regions]
        forward_in = intersect(region_nodes, active_forward)
        outside_fwd = setdiff(region_nodes, active_forward)

        # amount to overlap
        k_overlap = round(Int, overlap_fraction * length(forward_in))
        k_overlap = min(k_overlap, length(forward_in))

        overlap_nodes =
            k_overlap > 0 ?
            forward_in[randperm(length(forward_in))[1:k_overlap]] :
            Int[]
        union!(active_backward, overlap_nodes)

        # ensure backward flow has ~same size as forward
        k_total = max(1, round(Int, frac * length(region_nodes)))
        k_rest = max(0, k_total - length(overlap_nodes))
        k_rest = min(k_rest, length(outside_fwd))

        rest_nodes =
            k_rest > 0 ?
            outside_fwd[randperm(length(outside_fwd))[1:k_rest]] :
            Int[]
        union!(active_backward, rest_nodes)
    end

    return active_backward
end

###############################################################
#            FORWARD / BACKWARD FLOW PROPAGATION
###############################################################

function propagate_flows(node_dict, edge_dict, active_nodes::Set{Int}, flow_type::Symbol, num_outcomes)
    for node_id in active_nodes
        node = node_dict[node_id]
        vec = flow_type == :fwd ? :entropy_fwd : :entropy_bwd

        # Initialize if empty
        if isempty(getfield(node, vec))
            setfield!(node, vec, fill(log(num_outcomes)+1e-6, num_outcomes))
        end

        # Propagate along outgoing edges
        for eid in node.outgoing
            tgt = node_dict[edge_dict[eid].node2]
            tvec = getfield(tgt, vec)
            if isempty(tvec)
                setfield!(tgt, vec, zeros(num_outcomes))
            end
            # simple diffusion model
            getfield(tgt, vec)[:] .+= 0.1 .* getfield(node, vec)
        end
    end
end

###############################################################
#            TRUE GERSTENHABER BRACKET
###############################################################

"""
Gerstenhaber bracket for two entropy vectors f and g:

    [f,g]_i = Σ_j ( f_i * ∂g_j/∂x_i  -  g_i * ∂f_j/∂x_i )

Approximation:
We use discrete differences: ∂g_j/∂x_i ≈ (g_j - g_i)

Final formula:

    bracket[i] = Σ_j ( f[i]*(g[j] - g[i]) - g[i]*(f[j] - f[i]) )
"""
function gerstenhaber_bracket(f::Vector{Float64}, g::Vector{Float64})
    n = length(f)
    B = zeros(Float64, n)

    for i in 1:n
        Fi = f[i]
        Gi = g[i]
        accum = 0.0
        for j in 1:n
            accum += Fi*(g[j] - Gi) - Gi*(f[j] - Fi)
        end
        B[i] = accum
    end

    return B
end


###############################################################
#            COMPUTE OVERLAP ENTROPY (BRACKET)
###############################################################

function compute_overlap(node_dict, num_outcomes)
    for node in values(node_dict)
        # Ensure lengths are correct
        if length(node.entropy_fwd) != num_outcomes ||
           length(node.entropy_bwd) != num_outcomes

            @warn "Node $(node.id) entropy vectors inconsistent length, fixing."
            node.entropy_fwd = resize!(node.entropy_fwd, num_outcomes)
            node.entropy_bwd = resize!(node.entropy_bwd, num_outcomes)
        end

        # Compute overlap safely
        node.entropy_overlap = node.entropy_fwd .+ node.entropy_bwd
    end
end

###############################################################
#           CONSERVE TOTAL ENTROPY
###############################################################

function conserve_total_entropy(node_dict)
    all_vals = []

    for node in values(node_dict)
        append!(all_vals, node.entropy_fwd)
        append!(all_vals, node.entropy_bwd)
    end

    total = sum(all_vals)
    if total == 0
        return
    end

    # normalize all entropies so sum remains 1
    for node in values(node_dict)
        node.entropy_fwd ./= total
        node.entropy_bwd ./= total
    end
end

###############################################################
#           REGION-LEVEL TIME SERIES
###############################################################

function record_region_ts(node_dict, region_fractions)
    ts_fwd = Dict{String,Float64}()
    ts_bwd = Dict{String,Float64}()
    ts_ov  = Dict{String,Float64}()

    for region in keys(region_fractions)
        nodes = [n for n in values(node_dict) if region in n.regions]
        ts_fwd[region] = sum(sum(n.entropy_fwd) for n in nodes)
        ts_bwd[region] = sum(sum(n.entropy_bwd) for n in nodes)
        ts_ov[region]  = sum(sum(n.entropy_overlap) for n in nodes)
    end

    return ts_fwd, ts_bwd, ts_ov
end


function initialize_entropy!(node_dict, num_outcomes)
    for node in values(node_dict)
        node.entropy_fwd      = zeros(num_outcomes)
        node.entropy_bwd      = zeros(num_outcomes)
        node.entropy_overlap  = zeros(num_outcomes)
    end
end



###############################################################
#                   MAIN SIMULATION
###############################################################

num_outcomes = 4
num_steps = 50

# Initialize entropy 
initialize_entropy!(node_dict, num_outcomes)
region_fwd_ts = Dict(region => Float64[] for region in keys(region_fractions))
region_bwd_ts = Dict(region => Float64[] for region in keys(region_fractions))
region_ov_ts  = Dict(region => Float64[] for region in keys(region_fractions))

println("Beginning simulation...")
for t in 1:num_steps
    # choose forward/backward participants
    active_fwd = select_active_nodes(node_dict, region_fractions)
    active_bwd = select_backward_nodes(node_dict, region_fractions, active_fwd, 0.3)

    propagate_flows(node_dict, edge_dict, active_fwd, :fwd, num_outcomes)
    propagate_flows(node_dict, edge_dict, active_bwd, :bwd, num_outcomes)

    # combine by true Gerstenhaber bracket
    compute_overlap(node_dict, num_outcomes)

    # global conservation
    conserve_total_entropy(node_dict)

    # record
    ts_fwd, ts_bwd, ts_ov = record_region_ts(node_dict, region_fractions)

    for region in keys(region_fractions)
        push!(region_fwd_ts[region], ts_fwd[region])
        push!(region_bwd_ts[region], ts_bwd[region])
        push!(region_ov_ts[region],  ts_ov[region])
    end
end

###############################################################
#                     PLOTS
###############################################################

plt = plot(title="Forward / Backward / Overlap Entropy by Region",
           xlabel="Time", ylabel="Entropy")

for region in keys(region_fractions)
    plot!(1:num_steps, region_fwd_ts[region], label="$region forward", lw=2, linestyle=:solid)
    plot!(1:num_steps, region_bwd_ts[region], label="$region backward", lw=2, linestyle=:dash)
    plot!(1:num_steps, region_ov_ts[region],  label="$region overlap", lw=2, linestyle=:dot)
end

display(plt)

println("Simulation complete.")

