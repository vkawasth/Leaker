# entropy_flow.jl
# Julia 1.8+ recommended
using CSV, DataFrames, LinearAlgebra, SparseArrays, Random, Statistics

# Optional plotting dependencies:
# using Plots

# ------------------------
# Utilities: CSV loaders
# ------------------------
function load_nodes(path_or_io)
    df = CSV.read(path_or_io, DataFrame; delim=';',header=true,ignorerepeated=true)
    # Expect columns: id;pos_x;pos_y;pos_z;degree;isAtSampleBorder;regions
    # Normalize column names
    rename!(df, names(df) .=> lowercase.(String.(names(df))))
    if "id" ∉ names(df)
        error("Nodes CSV must contain column 'id'")
    end
    return df
end

function load_edges(path_or_io)
    df = CSV.read(path_or_io, DataFrame; delim=';',header=true,ignorerepeated=true)
    rename!(df, names(df) .=> lowercase.(String.(names(df))))
    # Expect at least node1id and node2id (or node1_id etc). Normalize known names
    if "node1id" ∈ names(df)
        rename!(df, :node1id => :n1, :node2id => :n2)
    elseif "node1_id" ∈ names(df)
        rename!(df, :node1_id => :n1, :node2_id => :n2)
    elseif ("node1" ∈ names(df)) && ("node2" ∈ names(df))
        rename!(df, :node1 => :n1, :node2 => :n2)
    else
        error("Edges CSV must contain node1id/node2id (or node1/node2)")
    end
    return df
end

# ------------------------
# Build index mapping
# ------------------------
"""
build_index_mapping(nodes_df, edges_df)

Returns (id2idx, idx2id, n)
- id2idx: Dict mapping original node id -> 1..n
- idx2id: vector of original ids
- n: number of nodes used
Any nodes referenced in edges but not present in nodes_df are added.
"""
function build_index_mapping(nodes_df, edges_df)
    ids_nodes = Vector{Int}(nodes_df.id)
    ids_edges = unique(vcat(Int.(edges_df.n1), Int.(edges_df.n2)))
    all_ids = unique(vcat(ids_nodes, ids_edges))
    sort!(all_ids)
    id2idx = Dict{Int,Int}()
    for (i,id) in enumerate(all_ids)
        id2idx[id] = i
    end
    idx2id = all_ids
    return id2idx, idx2id, length(all_ids)
end

# ------------------------
# Build adjacency / Laplacian
# ------------------------
"""
build_graph_matrices(edges_df, id2idx; weight_col=:length)

Returns adjacency (sparse) and graph Laplacian L (sparse).
Edges treated undirected; weight default is 1/length if length exists, else 1.
"""
function build_graph_matrices(edges_df, id2idx; weight_col=:length)
    m = length(id2idx)
    row = Int[]
    col = Int[]
    vals = Float64[]
    for r in eachrow(edges_df)
        # require both node id columns exist in this row
        if !(haskey(r, :n1) && haskey(r, :n2))
            continue
        end
        # read raw node ids (may throw if not parseable to Int)
        n1id = try
            Int(r[:n1])
        catch
            continue
        end
        n2id = try
            Int(r[:n2])
        catch
            continue
        end

        # ensure these ids are in the mapping
        if !(haskey(id2idx, n1id) && haskey(id2idx, n2id))
            continue
        end

        u = id2idx[n1id]
        v = id2idx[n2id]

        # default weight
        w = 1.0

        # if the weight column exists and is not missing, try to use it (inverse-length convention)
        if haskey(r, weight_col) && !ismissing(r[weight_col])
            sval = string(r[weight_col])
            length_val = tryparse(Float64, sval)
            if length_val !== nothing && length_val > 0.0
                w = 1.0 / length_val
            end
        end

        # push symmetric entries (undirected graph)
        push!(row, u); push!(col, v); push!(vals, w)
        push!(row, v); push!(col, u); push!(vals, w)
    end
    A = sparse(row,col,vals, m, m)
    degs = vec(sum(A,dims=2))
    D = spdiagm(0 => degs)
    L = D - A
    return A, L
end

# ------------------------
# Entropy / gradient / mobility / Jacobian
# ------------------------
"""
compute_entropy_gradient(p, pi)
returns grad F (vector) for KL: grad_i = log(p_i/pi_i) + 1
"""
function compute_entropy_gradient(p::Vector{Float64}, pi::Vector{Float64})
    return log.(p ./ pi) .+ 1.0
end

"""
hessian_entropy(p)
returns diagonal Hessian diag(1/p_i)
"""
hessian_entropy(p::Vector{Float64}) = Diagonal(1.0 ./ p)

"""
mobility_diag(p)
default mobility M = diag(p)
"""
mobility_diag(p::Vector{Float64}) = Diagonal(p)

"""
mobility_graph_laplacian(p, A)
builds a mobility of Laplacian form:
(M φ)_i = sum_j K_ij (φ_i - φ_j) ; we return M as sparse matrix
we use K_ij = w_ij * psi(p_i,p_j) with psi = (p_i + p_j)/2 (example)
"""
function mobility_graph_laplacian(p::Vector{Float64}, A::SparseMatrixCSC{Float64,Int})
    # Build K_ij from A entries
    m = size(A,1)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for colind in 1:size(A,2)
        for ptr in A.colptr[colind]:(A.colptr[colind+1]-1)
            rowind = A.rowval[ptr]
            w = A.nzval[ptr]
            if rowind < colind
                # process each pair once
                K = w * ((p[rowind] + p[colind]) / 2.0)
                # Laplacian contribution
                push!(rows, rowind); push!(cols, rowind); push!(vals, K)
                push!(rows, colind); push!(cols, colind); push!(vals, K)
                push!(rows, rowind); push!(cols, colind); push!(vals, -K)
                push!(rows, colind); push!(cols, rowind); push!(vals, -K)
            end
        end
    end
    M = sparse(rows, cols, vals, m, m)
    return M
end

"""
build_jacobian(p, M::AbstractMatrix, π)
Returns J = - M(p) * Hess(F)(p)
"""
function build_jacobian(p::Vector{Float64}, M::AbstractMatrix, pi::Vector{Float64})
    H = hessian_entropy(p)
    # If M is diagonal or full matrix, multiply
    return - (M * Matrix(H))
end

# ------------------------
# Nullspace / kernel detection via SVD
# ------------------------
"""
numeric_nullspace(A; tol=1e-8)

Returns matrix whose columns form an approximate basis for nullspace of A.
"""
function numeric_nullspace(A::AbstractMatrix; tol = 1e-8)
    U, S, Vt = svd(A)
    svals = diag(S)
    # tolerance relative to largest singular value
    smax = maximum(svals)
    thresh = max(tol, smax * 1e-8)
    cols = findall(x -> x ≤ thresh, svals)
    if isempty(cols)
        return zeros(size(A,2),0)
    else
        # Vt is transposed; columns of V corresponding to small svals are nullspace basis
        return Vt[:, cols]
    end
end

# ------------------------
# Time stepping: entropy gradient flow with optional partial node updates
# ------------------------
"""
entropy_flow_step!(p, pi, A, Mtype; dt=0.01, update_fraction=1.0)

Takes one explicit Euler step updating only a random subset of nodes
(update_fraction in (0,1]), returns new p (in-place).
"""
function entropy_flow_step!(p::Vector{Float64}, pi::Vector{Float64}, A::SparseMatrixCSC{Float64,Int};
                            dt=1e-3, Mtype=:diag, update_fraction=1.0)
    m = length(p)
    # Build mobility at current p:
    if Mtype == :diag
        M = mobility_diag(p)
    elseif Mtype == :laplacian
        M = mobility_graph_laplacian(p, A)
    else
        error("Unknown Mtype: $Mtype")
    end
    gradF = compute_entropy_gradient(p, pi)
    dp = - M * gradF  # note M * grad is vector (if M is matrix)
    # choose subset
    k = max(1, round(Int, update_fraction * m))
    idxs = (update_fraction >= 1.0) ? collect(1:m) : randperm(m)[1:k]
    # apply updates only to idxs (explicit Euler)
    for i in idxs
        p[i] += dt * dp[i]
    end
    # ensure positivity and renormalize to simplex
    for i in 1:m
        if p[i] < 1e-12
            p[i] = 1e-12
        end
    end
    p ./= sum(p)
    return p
end

# ------------------------
# Top-level runner: reads CSVs, builds problem, simulates, prints kernels
# ------------------------
function run_entropy_sim(nodes_path, edges_path; pi_mode=:uniform, mobility=:diag,
                         dt=1e-3, steps=1000, update_fraction=0.05, verbose=true)

    nodes = load_nodes(nodes_path)
    edges = load_edges(edges_path)
    id2idx, idx2id, n = build_index_mapping(nodes, edges)
    A, L = build_graph_matrices(edges, id2idx)
    println("N nodes = $n; adjacency nnz = $(nnz(A)); Laplacian nnz = $(nnz(L))")

    # initial p: small random perturbation around uniform, normalized
    p = rand(n); p ./= sum(p)
    if pi_mode == :uniform
        pi = fill(1.0/n, n)
    elseif pi_mode == :degree
        degs = vec(sum(A,dims=2))
        if sum(degs) == 0
            pi = fill(1.0/n, n)
        else
            pi = degs ./ sum(degs)
        end
    else
        error("pi_mode must be :uniform or :degree")
    end

    # build Markov generator (probability-model) as reference L_gen = -L_norm
    # Build a simple generator from adjacency: row-normalize adjacency -> transition matrix P, then L_gen = P - I
    degs = vec(sum(A,dims=2))
    P = spzeros(n,n)
    for j in 1:n
        if degs[j] > 0
            # get neighbors
            for ptr in A.colptr[j]:(A.colptr[j+1]-1)
                i = A.rowval[ptr]
                P[i,j] = A.nzval[ptr] / degs[j]
            end
        end
    end
    L_gen = P - I # column-stochastic P; L_gen * p corresponds to discrete generator

    # compute eigen / nullspace of L_gen as "probability-model kernel"
    try
        null_L = numeric_nullspace(Matrix(L_gen); tol=1e-8)
        println("probability-generator nullspace dimension = $(size(null_L,2))")
    catch e
        @warn "Failed computing nullspace of L_gen: $e"
    end

    # Simulation loop
    for t in 1:steps
        entropy_flow_step!(p, pi, A; dt=dt, Mtype=(mobility==:diag ? :diag : :laplacian),
                            update_fraction=update_fraction)
        if verbose && t % max(1,steps ÷ 10) == 0
            println("step $t  sum(p)=$(sum(p))  min(p)=$(minimum(p))  max(p)=$(maximum(p))")
        end
    end

    # compute Jacobian at final p
    if mobility == :diag
        M_final = mobility_diag(p)
    else
        M_final = mobility_graph_laplacian(p, A)
    end
    J = build_jacobian(p, M_final, pi)

    # numeric nullspace of J
    nullJ = numeric_nullspace(Matrix(J); tol=1e-8)
    println("Jacobian J size = $(size(J)); nullspace dimension = $(size(nullJ,2))")

    # return a dictionary of results
    return Dict(
        :nodes => nodes,
        :edges => edges,
        :A => A,
        :L => L,
        :p_final => p,
        :pi => pi,
        :J => J,
        :nullJ => nullJ,
        :nullL => get(null_L, nothing)
    )
end

# ------------------------
# Example usage
# ------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    # Replace these filenames with your paths or use the file-like string reading
    nodes_file = "/Users/vaw1/Downloads/OGB/node_regions_clean.csv"   # put your node table as semicolon CSV here
    edges_file = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"   # put your edges table as semicolon CSV here

    println("Calling run_entropy_sim with update_fraction = 0.05 (5%)")
    res = run_entropy_sim(nodes_file, edges_file; pi_mode=:degree, mobility=:diag,
                          dt=1e-3, steps=500, update_fraction=0.05, verbose=true)

    println("Final p summary: min=$(minimum(res[:p_final])), max=$(maximum(res[:p_final]))")
    println("Nullspace dimension of Jacobian J = $(size(res[:nullJ],2))")
    if size(res[:nullJ],2) > 0
        println("Basis vectors (columns) of nullspace (first 5 entries):")
        show(res[:nullJ][1:min(5,size(res[:nullJ],1)), :])
    end
    println("Done.")
end
