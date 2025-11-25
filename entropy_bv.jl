# entropy_bv.jl
# Minimal, practical Gerstenhaber (C0/C1/C2) + BV (Δ) support for large sparse graphs.
# Target: operate on an "active subgraph" (compact indices 1..m) for efficiency.
# Usage: include("entropy_bv.jl"); then use build_active_subgraph(...) and integrate_BV_correction!()

module EntropyBV

using SparseArrays, LinearAlgebra

export C0, C1, C2,
       build_active_subgraph, build_edge_index,
       c1_from_edgevals, c1_to_sparse, c1_commutator_bracket,
       cup_c1_c1_to_c2, delta_BV_C2_to_C1,
       derived_bracket_from_Delta,
       compute_BV_correction!, integrate_BV_correction!

# ---------------------------
# Types: compact, memory-friendly
# ---------------------------
"""
C0: 0-cochain as dense vector length = m (local active nodes)
"""
struct C0
    vals::Vector{Float64}   # length m
end

"""
C1: 1-cochain stored as edge-indexed arrays
- u::Vector{Int}, v::Vector{Int}   oriented endpoints in local indexing 1..m
- vals::Vector{Float64}            value per oriented edge (same length)
- idxmap::Dict{Tuple{Int,Int},Int} optional canonical mapping from (u,v) -> index
"""
struct C1
    u::Vector{Int}
    v::Vector{Int}
    vals::Vector{Float64}
    idxmap::Dict{Tuple{Int,Int},Int}
end

"""
C2: 2-cochain stored as path-indexed arrays for paths u->v->w
- u::Vector{Int}, v::Vector{Int}, w::Vector{Int}
- vals::Vector{Float64}
- idxmap::Dict{Tuple{Int,Int,Int},Int}
"""
struct C2
    u::Vector{Int}
    v::Vector{Int}
    w::Vector{Int}
    vals::Vector{Float64}
    idxmap::Dict{Tuple{Int,Int,Int},Int}
end

# ---------------------------
# Build active subgraph utilities
# ---------------------------
"""
build_active_subgraph(active_nodes::Vector{Int}, A::SparseMatrixCSC)
Return (id2local::Dict{Int,Int}, local2id::Vector{Int}, A_sub::SparseMatrixCSC)
active_nodes are global node indices (sorted or unsorted). A is global adjacency (n×n).
"""
function build_active_subgraph(active_nodes::Vector{Int}, A::SparseMatrixCSC)
    id2local = Dict{Int,Int}()
    for (i,id) in enumerate(active_nodes)
        id2local[id] = i
    end
    m = length(active_nodes)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for global_col in active_nodes
        col_local = id2local[global_col]
        for ptr in A.colptr[global_col]:(A.colptr[global_col+1]-1)
            global_row = A.rowval[ptr]
            if (haskey(id2local, global_row))
                row_local = id2local[global_row]
                push!(rows, row_local)
                push!(cols, col_local)
                push!(vals, A.nzval[ptr])
            end
        end
    end
    A_sub = sparse(rows, cols, vals, m, m)
    return id2local, active_nodes, A_sub
end

# Build compact edge list with index mapping (oriented edges)
"""
build_edge_index(A_sub::SparseMatrixCSC)
Returns (u,v,edge_idxmap)
- u[k], v[k] are endpoints in local indexing for oriented edge k (k=1:ne)
- idxmap[(u,v)] -> k
"""
function build_edge_index(A_sub::SparseMatrixCSC)
    rows, cols, vals = findnz(A_sub)
    ne = length(rows)
    u = Vector{Int}(undef, ne)
    v = Vector{Int}(undef, ne)
    idxmap = Dict{Tuple{Int,Int},Int}()
    for (k,(r,c)) in enumerate(zip(rows, cols))
        u[k] = r
        v[k] = c
        idxmap[(r,c)] = k
    end
    return u, v, idxmap
end

# ---------------------------
# Construct C1 from edge values (aligned with edge index)
# ---------------------------
"""
c1_from_edgevals(u,v,idxmap, vals_edge)
vals_edge Vector length = ne; builds C1
"""
function c1_from_edgevals(u::Vector{Int}, v::Vector{Int}, idxmap::Dict{Tuple{Int,Int},Int}, vals_edge::Vector{Float64})
    return C1(copy(u), copy(v), copy(vals_edge), deepcopy(idxmap))
end

"""
c1_to_sparse(c::C1, m) -> SparseMatrixCSC(m,m)
"""
function c1_to_sparse(c::C1, m::Int)
    return sparse(c.u, c.v, c.vals, m, m)
end

# ---------------------------
# C1-C1 Gerstenhaber bracket = commutator of operators (full support)
# implementation uses sparse matrices for speed
# ---------------------------
"""
c1_commutator_bracket(a::C1, b::C1, m::Int) -> C1 (their Gerstenhaber bracket)
Computes H = A*B - B*A where A,B are sparse matrices (C1 -> linear operators).
Returns C1 with same idx ordering as A and B unioned (we build new idxmap).
"""
function c1_commutator_bracket(a::C1, b::C1, m::Int)
    A = c1_to_sparse(a, m)
    B = c1_to_sparse(b, m)
    H = A*B - B*A
    rows, cols, vals = findnz(H)
    ne = length(rows)
    u = copy(rows); v = copy(cols)
    idxmap = Dict{Tuple{Int,Int},Int}()
    for k in 1:ne
        idxmap[(u[k], v[k])] = k
    end
    return C1(u, v, vals, idxmap)
end

# ---------------------------
# Cup: C1 ∪ C1 -> C2 (paths of length 2)
# For each oriented pair e1: u->v and e2: v->w produce path (u,v,w) with val = val1 * val2
# ---------------------------
"""
cup_c1_c1_to_c2(a::C1, b::C1)
Return C2 representing oriented length-2 paths.
Complexity O(ne + sum_outdeg^2) in worst case but practically O(ne + number of length-2 paths).
"""
function cup_c1_c1_to_c2(a::C1, b::C1)
    # Build outgoing index lists for a and b keyed by middle node v
    m = maximum(vcat(a.u, a.v, b.u, b.v))  # local m guess; not strictly needed
    # Outgoing edges from node x in a: edges_a_out[x] = list of indices k where a.u[k] == x
    edges_a_out = Vector{Vector{Int}}(undef, m)
    edges_b_out = Vector{Vector{Int}}(undef, m)
    for i in 1:m
        edges_a_out[i] = Int[]
        edges_b_out[i] = Int[]
    end
    for (k,mid) in enumerate(a.v)
        push!(edges_a_out[mid], k)
    end
    for (k,mid) in enumerate(b.u)
        push!(edges_b_out[mid], k)  # note: b.u is source of b (middle must match)
    end

    # Now iterate middle nodes where both lists non-empty
    u_list = Int[]; v_list = Int[]; w_list = Int[]; vals = Float64[]
    idxmap = Dict{Tuple{Int,Int,Int},Int}()
    kcount = 0
    for mid in 1:m
        list1 = edges_a_out[mid]
        list2 = edges_b_out[mid]
        if isempty(list1) || isempty(list2)
            continue
        end
        # combine
        for e1 in list1
            u = a.u[e1]; val1 = a.vals[e1]
            for e2 in list2
                w = b.v[e2]; val2 = b.vals[e2]
                kcount += 1
                push!(u_list, u); push!(v_list, mid); push!(w_list, w)
                push!(vals, val1 * val2)
                idxmap[(u, mid, w)] = kcount
            end
        end
    end
    return C2(u_list, v_list, w_list, vals, idxmap)
end

# ---------------------------
# BV operator Δ : C2 -> C1 by contracting the middle vertex
# - We map path (u -> v -> w) to oriented edge (u -> w) if that oriented edge exists in A_sub
# - weight contribution: path_val * opt_weight where opt_weight can be adjacency(u,w) or 1.0
# - If (u,w) not an existing oriented edge, we still optionally create it (here we create it to keep closure)
# ---------------------------
"""
delta_BV_C2_to_C1(c2::C2, A_sub::SparseMatrixCSC; use_edge_weight=true)
Returns C1. If use_edge_weight, contributions multiplied by A_sub[u,w] (0 if absent).
If you prefer to always create (u->w) even if not in A_sub, pass use_edge_weight=false.
"""
function delta_BV_C2_to_C1(c2::C2, A_sub::SparseMatrixCSC; use_edge_weight::Bool=true)
    # Build accumulator dict for oriented edge -> value
    acc = Dict{Tuple{Int,Int},Float64}()
    for (k, (u,v,w)) in enumerate(zip(c2.u, c2.v, c2.w))
        val = c2.vals[k]
        if use_edge_weight
            # check adjacency u->w
            wuw = getindex(A_sub, u, w)
            if wuw == 0.0
                continue
            end
            contrib = val * wuw
        else
            contrib = val
        end
        key = (u, w)
        acc[key] = get(acc, key, 0.0) + contrib
    end
    # unpack to arrays
    ne = length(acc)
    u_arr = Vector{Int}(undef, ne); v_arr = Vector{Int}(undef, ne); vals = Vector{Float64}(undef, ne)
    idxmap = Dict{Tuple{Int,Int},Int}()
    k = 0
    for ((uu,vv), vvval) in acc
        k += 1
        u_arr[k] = uu; v_arr[k] = vv; vals[k] = vvval
        idxmap[(uu,vv)] = k
    end
    return C1(u_arr, v_arr, vals, idxmap)
end

# ---------------------------
# Derived Gerstenhaber bracket from Δ:
# {a,b} = (-1)^{|a|} ( Δ(a∪b) - (Δ a) ∪ b - (-1)^{|a|} a ∪ (Δ b) )
# We implement for degrees up to 1/2 (C0/C1 combos) and full C1-C1 path route:
# For C1-C1:
# - a ∪ b -> C2 via cup_c1_c1_to_c2
# - Δ(a∪b) -> C1 via delta_BV_C2_to_C1
# - Δ a -> (C0 or C1?) For a C1, Δ(a) is undefined (Δ: C2->C1), so Δa = 0
# So bracket simplifies to {a,b} = (-1)^{1} Δ(a∪b) - 0 - (-1)^1 a∪(Δ b) -> but Δ b = 0 -> {a,b} = -Δ(a∪b)
# However Hochschild theoretic bracket should equal commutator. We implement both:
#  - derived_bracket_from_Delta_C1C1 returns Δ(a∪b) (signed) as a candidate
#  - c1_commutator_bracket computes the true Hochschild bracket (commutator)
# Use either for tests/consistency.
# ---------------------------
"""
derived_bracket_from_Delta(a::C1, b::C1, A_sub::SparseMatrixCSC)
Return C1 given by (-1)^{|a|} Δ(a∪b) - ... (for C1-C1 this reduces essentially to -Δ(a∪b))
"""
#=
function derived_bracket_from_Delta(a::C1, b::C1, A_sub::SparseMatrixCSC)
    # a,b degree 1 => (-1)^1 = -1
    c2 = cup_c1_c1_to_c2(a, b)
    delta = delta_BV_C2_to_C1(c2, A_sub; use_edge_weight=true)
    # sign = (-1)^1 = -1, return - delta (C1)
    delta_neg = copy(delta)
    delta_neg.vals .= .-delta_neg.vals
    return delta_neg
end

# A unified derived-bracket function that attempts types C0/C1 combos (we focus on C1-C1)
function derived_bracket_from_Delta_general(a::Abstract, b::Abstract, A_sub::SparseMatrixCSC)
    if isa(a, C1) && isa(b, C1)
        return derived_bracket_from_Delta(a,b,A_sub)
    else
        error("Derived bracket generalization not implemented for these types in this module.")
    end
end
=#
# ---------------------------
# Convert C1 into node-vector correction via divergence-like map:
# Map C1 (edges oriented u->v) into node vector of length m
# e.g., contribution to node i = sum_incoming edges vals_in - sum_outgoing edges vals_out (choose convention)
# ---------------------------
"""
c1_to_node_correction(c::C1, m::Int; convention=:in_minus_out)
Return C0 with correction per node.
"""

function c1_to_node_correction(c::C1, m::Int; convention=:in_minus_out)
    correction = zeros(Float64, m)
    @inbounds for k in eachindex(c.vals)
        u = c.u[k]
        v = c.v[k]
        val = c.vals[k]
        if convention === :in_minus_out
            correction[v] += val
            correction[u] -= val
        else
            correction[u] += val
            correction[v] -= val
        end
    end
    return correction::Vector{Float64}  # force return type
end
#=
function c1_to_node_correction(c::C1, m::Int; convention::Symbol=:in_minus_out)
    out = zeros(Float64, m)
    if convention == :in_minus_out
        for k in 1:length(c.vals)
            uu = c.u[k]; vv = c.v[k]; val = c.vals[k]
            out[vv] += val
            out[uu] -= val
        end
    elseif convention == :in_plus_out
        for k in 1:length(c.vals)
            uu = c.u[k]; vv = c.v[k]; val = c.vals[k]
            out[vv] += val
            out[uu] += val
        end
    else
        error("Unknown convention")
    end
    return C0(out)
end
=#

# ---------------------------
# High-level: compute BV-derived correction vector for p (C0) using the current flux as C1
# - p0: C0 (active nodes)
# - flux_c1: C1 representing current edge fluxes (e.g., K_ij = w*(p_i+p_j)/2)
# - A_sub: adjacency on active nodes
# - alpha: scaling factor
# Returns delta_p::C0 (length m) which you can add to dp (active nodes)
# ---------------------------
"""
compute_BV_correction!(delta_p::Vector{Float64}, p0::C0, flux::C1, A_sub::SparseMatrixCSC; alpha=1e-3)
Computes BV-derived correction and writes into delta_p (assumed length m). Does not normalize.
"""
function compute_BV_correction!(delta_p::Vector{Float64}, p0::C0, flux::C1, A_sub::SparseMatrixCSC; alpha::Float64=1e-3)
    # Build C1 from flux (flux is already C1)
    # Compute derived bracket (C1)
    db = derived_bracket_from_Delta_general(flux, flux, A_sub)  # using flux-flux as a simple self-bracket probe
    # Convert resulting C1 to node correction
    corr = c1_to_node_correction(db, length(p0.vals), convention=:in_minus_out)
    # Apply scaling
    delta_p .= delta_p .+ alpha .* corr.vals
    return nothing
end

"""
integrate_BV_correction!(p_active::Vector{Float64}, dp_active::Vector{Float64},
                          A_sub::SparseMatrixCSC, edge_u, edge_v;
                          alpha=1e-3, use_flux_builder=true)

Convenience wrapper: builds flux K_{ij} = w_{ij}*(p_i + p_j)/2 (using A_sub weights)
as C1 (aligned to edges in A_sub), computes BV correction and adds into dp_active in-place.
"""
function integrate_BV_correction!(p_active::Vector{Float64}, dp_active::Vector{Float64},
                                  A_sub::SparseMatrixCSC, edge_u::Vector{Int}, edge_v::Vector{Int};
                                  alpha::Float64=1e-3, use_edge_weight::Bool=true)

    m = length(p_active)
    # Build flux values matching oriented edges in edge_u/edge_v
    ne = length(edge_u)
    flux_vals = Vector{Float64}(undef, ne)
    for k in 1:ne
        ui = edge_u[k]; vi = edge_v[k]
        w = getindex(A_sub, ui, vi)
        flux_vals[k] = w * ((p_active[ui] + p_active[vi]) / 2.0)
    end
    # Build C1
    idxmap = Dict{Tuple{Int,Int},Int}()
    for k in 1:ne
        idxmap[(edge_u[k], edge_v[k])] = k
    end
    flux_c1 = C1(edge_u, edge_v, flux_vals, idxmap)

    # compute correction
    delta_p = zeros(Float64, m)
    compute_BV_correction!(delta_p, C0(p_active), flux_c1, A_sub; alpha=alpha)

    # integrate into dp_active
    dp_active .+= delta_p

    return nothing
end

# ---------------------------
# Small test
# ---------------------------
function _self_test()
    # small directed-style graph (local indices 1..4)
    rows = [1,2,2,3]; cols=[2,3,4,4]; vals=[1.0,1.0,1.0,1.0]
    A_sub = sparse(rows, cols, vals, 4, 4)
    edge_u, edge_v, idxmap = build_edge_index(A_sub)
    println("Edges:", zip(edge_u,edge_v))
    p = rand(4); p ./= sum(p)
    dp = zeros(4)
    println("p:", p)
    integrate_BV_correction!(p, dp, A_sub, edge_u, edge_v; alpha=1e-2)
    println("dp after BV correction:", dp)
    # verify bracket commutator approx equals derived delta up to differences (toy check)
    # build two random C1s
    ne = length(edge_u)
    randvals1 = rand(ne); randvals2 = rand(ne)
    c1a = C1(edge_u, edge_v, randvals1, Dict{Tuple{Int,Int},Int}())
    c1b = C1(edge_u, edge_v, randvals2, Dict{Tuple{Int,Int},Int}())
    comm = c1_commutator_bracket(c1a, c1b, 4)
    # cup->delta derived
    derived = derived_bracket_from_Delta(c1a, c1b, A_sub)
    println("comm ne=", length(comm.vals), " derived ne=", length(derived.vals))
    return true
end

# run test when module loaded directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running EntropyBV self-test...")
    _self_test()
end

#Force method replacement
#Base.delete_method.(methods(EntropyBV.c1_to_node_correction))
#@info "c1_to_node_correction RELOADED — returns Vector{Float64}"

end # module EntropyBV
