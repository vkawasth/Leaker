using CSV, DataFrames
using Catlab
using Catlab.CategoricalAlgebra
using Random

Random.seed!(42)  # reproducibility

# -------------------------
# 1️⃣ Load node and edge data
# -------------------------
NODES = "/Users/vaw1/Downloads/OGB/node_regions_clean.csv" 
EDGES = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"

nodes_df = CSV.read(NODES, DataFrame; delim=';')
edges_df = CSV.read(EDGES, DataFrame; delim=';')

# Only few will perturb across physical structures forming functional connectomes.
# Say 2000 in a region

#N = 2000  # or whatever number per region
#active_nodes = Dict{String, Vector{Int}}()

#for (r, nodes) in region_nodes
#    node_list = collect(nodes)
#    active_nodes[r] = randperm(length(node_list))[1:min(N, length(node_list))]
#    active_nodes[r] = node_list[active_nodes[r]]
#end

# Convert regions column from string to actual arrays
# -------------------------
# 1️⃣ Parse regions column safely
# -------------------------
function parse_region_list(s::AbstractString)
    # remove brackets and quotes
    s_clean = replace(s, r"[\[\]']" => "")
    # split by comma and remove extra spaces
    regions = strip.(split(s_clean, ","))
    return regions
end

nodes_df.regions = [parse_region_list(r) for r in nodes_df.regions]
region_nodes = Dict{String, Set{Int}}()
for row in eachrow(nodes_df)
    for r in row.regions
        if !haskey(region_nodes, r)
            region_nodes[r] = Set([row.id])
        else
            push!(region_nodes[r], row.id)
        end
    end
end

# -------------------------
# 2️⃣ Build regions as FinSets
# -------------------------
# Step 2: convert each Set to immutable FinSet
regions = Dict{String, FinSet}()

for (r, node_set) in region_nodes
    regions[r] = FinSet(collect(node_set))
end

# -------------------------
# 3️⃣ Select sparse active nodes per region
# -------------------------
sparsity_fraction = 0.1  # only 10% of nodes active
active_nodes = Dict{String, FinSet}()

for (r, nodeset) in regions
    n_active = Int(round(length(nodeset) * sparsity_fraction))
    active_nodes[r] = FinSet(rand(collect(nodeset), n_active))
end

# -------------------------
# 4️⃣ Build adjacency per region (only active nodes)
# -------------------------
adjacency = Dict{String, Dict{Int, Set{Int}}}()

for (r, nodeset) in active_nodes
    adj_r = Dict{Int, Set{Int}}()
    for node in nodeset
        # neighbors from edges_df that are also active in this region
        neighbors = Set([e.node2id for e in eachrow(edges_df)
                         if e.node1id == node && e.node2id in nodeset])
        adj_r[node] = neighbors
    end
    adjacency[r] = adj_r
end

# -------------------------
# 5️⃣ Initialize flows per region
# -------------------------
flows_per_region = 2
flow_maps = Dict{String, Dict{Int, Dict{Int, Float64}}}()

for (r, nodeset) in active_nodes
    flow_maps[r] = Dict()
    for f in 1:flows_per_region
        flow_maps[r][f] = Dict(node => rand() for node in nodeset)
    end
end

# -------------------------
# 6️⃣ Define flow propagation function
# -------------------------
function propagate_entropy!(flow_map::Dict{Int, Float64}, adjacency::Dict{Int, Set{Int}}; α=0.1)
    new_flow = Dict(node => flow_map[node] for node in keys(flow_map))
    for node in keys(flow_map)
        inflow = sum(flow_map[neighbor] for neighbor in adjacency[node])
        new_flow[node] += α * inflow
    end
    return new_flow
end

# -------------------------
# 7️⃣ Run time-varying simulation
# -------------------------
num_steps = 10

for t in 1:num_steps
    println("Step $t")
    for (r, _) in active_nodes
        for f in 1:flows_per_region
            flow_maps[r][f] = propagate_entropy!(flow_maps[r][f], adjacency[r])
        end
    end
end

# -------------------------
# 8️⃣ Combine flows across regions
# -------------------------
combined_flow = Dict{Int, Float64}()

for (r, flows) in flow_maps
    for f in 1:flows_per_region
        for (node, val) in flows[f]
            combined_flow[node] = get(combined_flow, node, 0.0) + val
        end
    end
end

println("Combined flow computed for $(length(combined_flow)) nodes")
