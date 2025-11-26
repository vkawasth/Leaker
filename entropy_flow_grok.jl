# entropy_flow_optimized.jl
# Julia 1.9+ recommended
using CSV, DataFrames, LinearAlgebra, SparseArrays, Random, Statistics
using Arpack          # for sparse eigs / nullspace
using IterativeSolvers # for lsqr, etc. if needed

# ------------------------
# Keep loaders unchanged (they're fine)
# ------------------------
function load_nodes(path_or_io)
    df = CSV.read(path_or_io, DataFrame; delim=';', header=true, ignorerepeated=true)
    rename!(df, names(df) .=> lowercase.(String.(names(df))))
    if !("id" in names(df))
        error("Nodes CSV must contain column 'id'")
    end
    return df
end

function load_edges(path_or_io)
    df = CSV.read(path_or_io, DataFrame; delim=';', header=true, ignorerepeated=true)
    rename!(df, names(df) .=> lowercase.(String.(names(df))))
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
# Index mapping (unchanged, fast enough)
# ------------------------
function build_index_mapping(nodes_df, edges_df)
    ids_nodes = Set(Int.(nodes_df.id))
    ids_edges = Set{Int}(vcat(edges_df.n1, edges_df.n2))
    all_ids = sort!(collect(union(ids_nodes, ids_edges)))
    id2idx = Dict(id => i for (i, id) in enumerate(all_ids))
    return id2idx, all_ids, length(all_ids)
end

# ------------------------
# Build adjacency (optimized, no string parsing per row)
# ------------------------
function build_graph_matrices(edges_df, id2idx; weight_col=:length)
    n = maximum(values(id2idx))
    I = Int[]; J = Int[]; V = Float64[]

    for row in eachrow(edges_df)
        n1 = get(row, :n1, missing)
        n2 = get(row, :n2, missing)
        ismissing(n1) || ismissing(n2) && continue

        u = get(id2idx, Int(n1), 0)
        v = get(id2idx, Int(n2), 0)
        (u == 0 || v == 0) && continue

        w = 1.0
        if haskey(row, weight_col) && !ismissing(row[weight_col])
            len = tryparse(Float64, string(row[weight_col]))
            if !isnothing(len) && len > 0
                w = 1.0 / len
            end
        end

        push!(I, u, v)
        push!(J, v, u)
        push!(V, w, w)
    end

    A = sparse(I, J, V, n, n)
    D = Diagonal(vec(sum(A; dims=2)))
    L = D - A
    return A, L
end

# ------------------------
# Entropy utilities (unchanged)
# ------------------------
compute_entropy_gradient(p, pi) = log.(p ./ pi) .+ 1.0
hessian_entropy(p) = Diagonal(1.0 ./ p)
mobility_diag(p) = Diagonal(p)

# ------------------------
# Critical fix: avoid building full mobility matrix
# We define matrix-free multiplication instead
# ------------------------
struct MobilityLaplacian{T<:AbstractVector} <: AbstractMatrix{Float64}
    p::T
    A::SparseMatrixCSC{Float64,Int}
end

Base.size(M::MobilityLaplacian) = (length(M.p), length(M.p))
Base.eltype(::Type{MobilityLaplacian{T}}) where T = Float64

function LinearAlgebra.mul!(y::Vector{Float64}, M::MobilityLaplacian, x::Vector{Float64})
    p, A = M.p, M.A
    fill!(y, 0.0)
    for j in 1:length(p)
        pj = p[j]
        xj = x[j]
        for idx in A.colptr[j]:(A.colptr[j+1]-1)
            i = A.rowval[idx]
            w = A.nzval[idx]
            pi = p[i]
            K = w * (pi + pj) / 2.0
            y[i] += K * (xj - x[i])
            y[j] += K * (x[i] - xj)  # symmetric
        end
    end
    return y
end

# Optional: also support y = M * x
function *(M::MobilityLaplacian, x::Vector{Float64})
    y = similar(x)
    mul!(y, M, x)
    return y
end

# ------------------------
# Jacobian as matrix-free operator
# J φ = - M * (H φ) where H = diag(1./p)
# → J φ = - M * (φ ./ p)
# ------------------------
# ————————————————————————
# Jacobian as matrix-free operator: J φ = -M (φ ./ p)
# ———————————————————————
struct JacobianOperator{T}
    p::T
    M::AbstractMatrix{Float64}   # Diagonal or MobilityLaplacian
end

Base.size(J::JacobianOperator) = (length(J.p), length(J.p))

function LinearAlgebra.mul!(y::Vector{Float64}, J::JacobianOperator, x::Vector{Float64})
    # J φ = -M (φ ./ p)
    # J φ = - M (φ ./ p)
    @inbounds @simd for i in eachindex(y)
        y[i] = x[i] / J.p[i]                 # temporary φ ./ p
    end
    mul!(y, J.M, y)                          # y ← M * (φ ./ p)
    lmul!(-1.0, y)                           # y ← -y   (this is the safe way)
    return y
end

*(J::JacobianOperator, x) = (y=similar(x); mul!(y,J,x); y)

# ------------------------
# Sparse nullspace via Arpack (only smallest eigenvalues)
# ------------------------
function sparse_nullspace(A::AbstractMatrix; nev=6, tol=1e-8)
    # We look for eigenvalues near zero
    λ, ϕ, nconv = eigs(A; nev=nev, which=:SM, tol=tol, maxiter=300)
    null_idx = findall(abs.(λ) .< max(1e-8, tol * maximum(abs.(λ))))
    if isempty(null_idx)
        return zeros(eltype(A), size(A,2), 0)
    else
        return ϕ[:, null_idx]
    end
end

# Fallback: if matrix-free, wrap in LinearMap
using LinearMaps
function sparse_nullspace_lm(Jop::JacobianOperator; nev=6, tol=1e-8)
    LM = LinearMap{Float64}(Jop, size(Jop,1); ismutating=false)
    λ, ϕ, nconv = eigs(LM; nev=nev, which=:SM, tol=tol, maxiter=500)
    null_idx = findall(abs.(λ) .< 1e-6)
    return isempty(null_idx) ? zeros(size(Jop,2), 0) : ϕ[:, null_idx]
end

# ------------------------
# Optimized time step (in-place, minimal allocations)
# ------------------------
function entropy_flow_step!(p::Vector{Float64}, pi::Vector{Float64}, A::SparseMatrixCSC;
                            dt=1e-3, Mtype=:diag, update_fraction=0.05)
    n = length(p)
    gradF = compute_entropy_gradient(p, pi)

    if Mtype === :diag
        # M(p) ∇F = p .* ∇F  ⇒  dp = -p .* ∇F
        dp = @muladd -p .* gradF                     # one allocation, very fast
    elseif Mtype === :laplacian
        mob = MobilityLaplacian(p, A)
        dp  = -mob * gradF                           # uses matrix-free mul!
    else
        error("Unknown Mtype")
    end

    # ───── random subset update ─────
    n = length(p)
    k = max(1, round(Int, update_fraction * n))
    idxs = update_fraction ≥ 1.0 ? (1:n) : randperm(n)[1:k]

    @inbounds for i in idxs
        p[i] += dt * dp[i]
        p[i] = max(p[i], 1e-15)
    end

    # Renormalize (stable)
    s = sum(p)
    @inbounds for i in 1:n
        p[i] /= s
    end

    return p
end

# ------------------------
# Main simulation (fully optimized)
# ------------------------
function run_entropy_sim(nodes_path, edges_path;
                         pi_mode=:degree,
                         mobility=:diag,
                         dt=1e-3,
                         steps=1000,
                         update_fraction=0.05,
                         verbose=true)

    nodes = load_nodes(nodes_path)
    edges = load_edges(edges_path)
    id2idx, idx2id, n = build_index_mapping(nodes, edges)
    A, L = build_graph_matrices(edges, id2idx)

    @info "Graph: $(n) nodes, $(nnz(A)÷2) undirected edges"

    p = fill(1.0/n, n)
    p .+= rand(n) .* 1e-6
    p ./= sum(p)

    # Target measure π
    if pi_mode === :uniform
        pi = fill(1.0/n, n)
    elseif pi_mode === :degree
        deg = vec(sum(A; dims=2))
        s = sum(deg)
        pi = s > 0 ? deg ./ s : fill(1.0/n, n)
    end

    # Simulation loop
    for it in 1:steps
        entropy_flow_step!(p, pi, A; dt=dt, Mtype=mobility, update_fraction=update_fraction)
        if verbose && (it % max(1, steps÷10) == 0 || it == steps)
            @info "Step $it: min(p)=$(minimum(p):.3e), max(p)=$(maximum(p):.3e)"
        end
    end

    # Build final Jacobian operator (matrix-free)
    M_final = mobility === :diag ? Diagonal(p) : MobilityLaplacian(p, A)
    Jop     = JacobianOperator(p, M_final)

    LM = LinearMap{Float64}( (y,x) -> mul!(y, Jop, x), length(p); ismutating=true)
    λ, ϕ, info = eigs(LM; nev=10, which=:SM, tol=1e-8, maxiter=500)

    null_dim = count(abs.(λ) .< 1e-6)
    nullJ    = null_dim == 0 ? zeros(length(p),0) : ϕ[:, abs.(λ) .< 1e-6]

    @info "Nullspace dimension = $null_dim  (out of $(length(p)) nodes)"

    return Dict(
        :p_final => p,
        :pi => pi,
        :A => A,
        :nullJ => nullJ,
        # now small matrix, e.g., n×10
    )
end

# ------------------------
# Run (example)
# ------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    nodes_file = "/Users/vaw1/Downloads/OGB/node_regions_clean.csv"   # put your node table as semicolon CSV here
    edges_file = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"   # put your edges table as semicolon CSV here

    res = run_entropy_sim(nodes_file, edges_file;
                          pi_mode = :degree,
                          mobility = :diag,        # use :laplacian only if you really need it
                          dt = 2e-3,
                          steps = 2000,
                          update_fraction = 0.1,
                          verbose = true)

    p = res[:p_final]
    @info "Final p: min=$(minimum(p)) max=$(maximum(p)) sum=$(sum(p))"
    @info "Nullspace dim = $(size(res[:nullJ], 2))"
end
