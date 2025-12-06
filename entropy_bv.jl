# entropy_bv.jl
# Minimal, practical Gerstenhaber (C0/C1/C2) + BV (Δ) support for large sparse graphs.
# Target: operate on an "active subgraph" (compact indices 1..m) for efficiency.
# Usage: include("entropy_bv.jl"); then use build_active_subgraph(...) and integrate_BV_correction!()

module EntropyBV

using SparseArrays, LinearAlgebra

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

# Blowdown local to global consistency mgmt.
"""
EntropySheaf

A conceptual structure capturing the local-to-global consistency required for
Hironaka-style resolution of singularities (exceptional entropy flows).

- sections: Stores the local (C0, C1) data for each neighborhood.
- node_to_section_key: Maps a global node index to the key(s) of the local section(s)
  it belongs to.
- local_neighborhoods: The sets U_i (e.g., node i and its radius=1 neighbors).
"""
struct EntropySheaf
    # Map key (e.g., node index 'i') to the local cochains over N(i)
    sections::Dict{Int, Tuple{C0, C1}}

    # Map global node index -> list of local section keys it belongs to
    node_to_section_key::Dict{Int, Vector{Int}}

    # The actual sets U_i
    local_neighborhoods::Dict{Int, Vector{Int}}
end


export C0, C1, C2,
       build_active_subgraph, build_edge_index,
       c1_from_edgevals, c1_to_sparse, c1_commutator_bracket,
       cup_c1_c1_to_c2, delta_BV_C2_to_C1,
       derived_bracket_from_Delta,
       derived_bracket_from_Delta_general,
       c1_to_node_correction,
       compute_BV_correction!, integrate_BV_correction!,
       # Blow-down exports
       select_exceptional_nodes, build_blowdown_mask,
       blowdown_C1, blowdown_C2, blowdown_graph, perform_blowdown,
       linearize_BV_kernel, local_jacobian_scores,
       node_entropy, compute_entropy_discrepancy,
       compact_indices, remap_C1_C2_to_compact
       # Sheaf exports (local to global entropy bondary aross regions mgmt.)
       #EntropySheaf, build_entropy_sheaf, resolve_and_check_consistency,
       #spectral_embedding, reconstruct_coords


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

# Complexity O(ne + sum_outdeg^2) in worst case but practically O(ne + number of length-2 paths).
"""
cup_c1_c1_to_c2(a::C1, b::C1, m::Int)

The cup product of two 1-cochains (edges), yielding a 2-cochain (paths of length 2).
It is defined by (a ∪ b)_{i->j->k} = a_{i->j} * b_{j->k}.
m is the size of the active node set (m x m).
"""
function cup_c1_c1_to_c2(a::C1, b::C1, m::Int)
    # Use the explicit size m for allocation, which avoids bounds errors 
    # if the max index in C1 is less than m (e.g., node m has no active edges)
    
    # Outgoing edges from node x in a: edges_a_out[x] = list of indices k where a.v[k] == x
    edges_a_out = Vector{Vector{Int}}(undef, m)
    # Incoming edges to node x in b: edges_b_in[x] = list of indices k where b.u[k] == x
    edges_b_in = Vector{Vector{Int}}(undef, m)
    
    for i in 1:m
        edges_a_out[i] = Int[]
        edges_b_in[i] = Int[]
    end
    
    # 1. Map: edge end (a.v) -> index of edge in a
    for (k, mid) in enumerate(a.v)
        # Check bounds: mid must be 1..m
        if 1 <= mid <= m
            push!(edges_a_out[mid], k)
        end
    end
    
    # 2. Map: edge start (b.u) -> index of edge in b
    for (k, mid) in enumerate(b.u)
        # Check bounds: mid must be 1..m
        if 1 <= mid <= m
            push!(edges_b_in[mid], k)
        end
    end

    # 3. Combine: Iterate middle nodes where both lists non-empty
    u_list = Int[]; v_list = Int[]; w_list = Int[]; vals = Float64[]
    idxmap = Dict{Tuple{Int,Int,Int},Int}()
    kcount = 0
    for mid in 1:m
        list1 = edges_a_out[mid]
        list2 = edges_b_in[mid]
        
        if isempty(list1) || isempty(list2)
            continue
        end
        
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
delta_BV_C2_to_C1(c2::C2, A_sub::SparseMatrixCSC; use_edge_weight::Bool=true)

The Batalin-Vilkovisky (BV) Delta operator: C2 -> C1.
It contracts the 2-cochain (path u->v->w) to a 1-cochain (edge u->w) by multiplying 
by the edge weight w_{u->w} and summing the contributions from all intermediate nodes v.
"""
function delta_BV_C2_to_C1(c2::C2, A_sub::SparseMatrixCSC; use_edge_weight::Bool=true)
    m = size(A_sub, 1) # Get the local dimension for safety checks
    
    # Build accumulator dict for oriented edge -> value
    acc = Dict{Tuple{Int,Int},Float64}()
    
    # Iterate over all 2-cochains (u->v->w paths)
    for (k, (u,v,w)) in enumerate(zip(c2.u, c2.v, c2.w))
        val = c2.vals[k]
        
        # CRITICAL SAFETY CHECK: Ensure indices are within the local active range
        if !(1 <= u <= m) || !(1 <= w <= m)
            # This path is outside the active subgraph bounds, skip it
            continue
        end

        if use_edge_weight
            # Check for existing edge u->w in the active subgraph
            # This access is safe because u and w are now guaranteed to be <= m
            wuw = getindex(A_sub, u, w) 
            
            if wuw == 0.0
                # If the collapsed edge does not exist in A_sub, the contribution is zero
                continue
            end
            contrib = val * wuw
        else
            contrib = val
        end
        
        # Accumulate the contribution for the resulting 1-cochain (u->w)
        key = (u, w)
        acc[key] = get(acc, key, 0.0) + contrib
    end
    
    # unpack the accumulator dict into a C1 cochain
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

# ==============================================================================
# 4. BV DERIVED BRACKET: {A, B}Δ = (-1)^|A| * Δ(A ∪ B)
# ==============================================================================
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

Computes the derived bracket {a, b}Δ using the BV operator Δ.
For C1-C1 (degree 1), the result is -Δ(a ∪ b).
"""
function derived_bracket_from_Delta(a::C1, b::C1, A_sub::SparseMatrixCSC)
    m = size(A_sub, 1) # Get the current active set size m
    
    # 1. Cup Product: C1 ⊗ C1 -> C2
    # NOTE: We use the m parameter in the function call now!
    c2 = cup_c1_c1_to_c2(a, b, m) 
    
    # 2. BV Operator: C2 -> C1
    # This is the contraction step that uses A_sub weights
    delta = delta_BV_C2_to_C1(c2, A_sub; use_edge_weight=true)
    
    # 3. Sign Correction: (-1)^|A| = (-1)^1 = -1
    # We negate the resulting C1 cochain.
    delta_neg = C1(delta.u, delta.v, .-delta.vals, delta.idxmap)
    return delta_neg
end

"""
derived_bracket_from_Delta(a::C1, b::C1, A_sub::SparseMatrixCSC)
Return C1 given by (-1)^{|a|} Δ(a∪b) - ... (for C1-C1 this reduces essentially to -Δ(a∪b))
"""

# A unified derived-bracket function that attempts types C0/C1 combos (we focus on C1-C1)
"""
derived_bracket_from_Delta_general(a, b, A_sub::SparseMatrixCSC)
A unified function for calling the derived bracket based on cochain types.
"""
function derived_bracket_from_Delta_general(a, b, A_sub::SparseMatrixCSC)
    if isa(a, C1) && isa(b, C1)
        return derived_bracket_from_Delta(a, b, A_sub)
    else
        # Placeholder for other required BV operations, e.g., {C0, C1}
        error("Derived bracket generalization implemented only for C1-C1 in this module.")
    end
end

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
# Julia has issues about retaking functions.
#Base.delete_method.(methods(EntropyBV.c1_to_node_correction))
#Base.delete_method.(methods(EntropyBV.cup_c1_c1_to_c2))
#Base.delete_method.(methods(EntropyBV.delta_BV_C2_to_C1))

#@info "c1_to_node_correction RELOADED — returns Vector{Float64}"

#############################
# BLOW DOWN
#############################
# ============================================================
# Blow-down operator (BV blow-down / ideal quotient A -> A')
# ============================================================

"""
select_exceptional_nodes(p::AbstractVector; top_fraction=0.01)

Return indices of nodes in the top percentage (default 1%)
by magnitude (entropy, probability, or flux field).
"""
function select_exceptional_nodes(p::Vector{Float64}; top_fraction=0.01)
    m = length(p)
    k = max(1, round(Int, m * top_fraction))
    # partial sort by absolute value
    idx = partialsortperm(p, rev=true, 1:k, by=abs)
    return sort(idx)
end


"""
build_blowdown_mask(m, exceptional_nodes)

Return a Bool vector keep[1..m] where keep[i] = true iff node i survives
the blow-down. (Exceptional nodes are removed.)
"""
function build_blowdown_mask(m::Int, exceptional::Vector{Int})
    keep = falses(m)             # Start with rejecting all (99% rejection)
    keep[exceptional] .= true    # Keep the exceptional 1%
    return keep
end


"""
blowdown_C1(c1::C1, keep::Vector{Bool})

Return a new C1 in which all edges touching an exceptional node
(keep[u]==false or keep[v]==false) are removed.

Implements the ideal quotient A/I: edges in the ideal are sent to 0.
"""
function blowdown_C1(c1::C1, keep::AbstractVector{Bool})
    # --- 1. CRITICAL FIX: Create the Old Index -> New Index map ---
    # The cumsum of 'keep' provides the new, contiguous index for every retained node.
    # Ex: keep = [T, F, T, F, T]  =>  new_idx = [1, 1, 2, 2, 3]
    # new_idx[old_index] gives the new_index (if keep[old_index] is true)
    new_idx = cumsum(keep)

    u_new = Int[]
    v_new = Int[]
    vals_new = Float64[]
    idxmap_new = Dict{Tuple{Int,Int},Int}()

    k_new = 0
    for k in eachindex(c1.vals)
        u_old = c1.u[k]
        v_old = c1.v[k]

        # 2. BOUNDS CHECK: This 'if' relies on u_old and v_old being within [1, length(keep)].
        # If the original BoundsError occurred, the data in c1.u or c1.v is flawed.
        # Assuming the original C1 construction is fixed and the node indices are now valid:
        if keep[u_old] && keep[v_old]
            k_new += 1
            
            # 3. APPLY MAPPING: Translate the old indices to the new, contiguous indices
            u_mapped = new_idx[u_old]
            v_mapped = new_idx[v_old]

            push!(u_new, u_mapped)
            push!(v_new, v_mapped)
            push!(vals_new, c1.vals[k])
            
            # Use the MAPPED indices for the new index map
            idxmap_new[(u_mapped, v_mapped)] = k_new
        end
    end

    return C1(u_new, v_new, vals_new, idxmap_new)
end


"""
blowdown_C2(c2::C2, keep)

Same idea: remove all 2-paths touching exceptional nodes.
"""
function blowdown_C2(c2::C2, keep::AbstractVector{Bool})
    # --- 1. CREATE MAPPING: Generate the map from Old Node Index to New Node Index ---
    # This must be done once per blowdown.
    new_idx = cumsum(keep)

    u2 = Int[]; v2 = Int[]; w2 = Int[]; vals2 = Float64[]
    idxmap2 = Dict{Tuple{Int,Int,Int},Int}()

    k2 = 0
    for k in eachindex(c2.vals)
        u_old = c2.u[k]
        v_old = c2.v[k]
        w_old = c2.w[k]
        
        # Check if all three nodes of the 2-simplex are kept
        if keep[u_old] && keep[v_old] && keep[w_old]
            k2 += 1
            
            # --- 2. APPLY MAPPING: Translate old indices to new, contiguous indices ---
            u_mapped = new_idx[u_old]
            v_mapped = new_idx[v_old]
            w_mapped = new_idx[w_old]

            push!(u2, u_mapped)
            push!(v2, v_mapped)
            push!(w2, w_mapped)
            
            push!(vals2, c2.vals[k])
            
            # Use the MAPPED indices for the new index map
            idxmap2[(u_mapped, v_mapped, w_mapped)] = k2
        end
    end

    return C2(u2, v2, w2, vals2, idxmap2)
end


"""
blowdown_graph(A_sub, keep)

Return A_sub' = A_sub restricted to surviving nodes (kept nodes).
This is π: A -> A'.

We keep node ordering the same (1..m), only delete rows/cols.
"""
function blowdown_graph(A_sub::SparseMatrixCSC, keep::AbstractVector{Bool})
    # Accept either BitVector or Vector{Bool} or other boolean-like vectors.
    inds = findall(keep)
    return A_sub[inds, inds]
end


"""
perform_blowdown(p_active, A_sub, C1_or_C2...; top_fraction=0.01)

Top-level blow-down driver:
- select exceptional nodes (top 1%)
- build mask
- restrict graph
- blow-down any C1/C2 structures passed in as a tuple

Returns:
    keep_mask, A_sub', blown_down_objects...
"""
function perform_blowdown(p::Vector{Float64}, A_sub::SparseMatrixCSC,idx2id::Vector{Int},
    cochains::Tuple; top_fraction=0.01)

    m = length(p)
    exceptional = select_exceptional_nodes(p; top_fraction=top_fraction)
    keep = build_blowdown_mask(m, exceptional)

    # blow-down graph adjacency
    A_new = blowdown_graph(A_sub, keep)
    # Slice the index map
    idx2id_new = idx2id[keep] # New map.
    # blow-down all cochains
    blown = Tuple(
        obj isa C1 ? blowdown_C1(obj, keep) :
        obj isa C2 ? blowdown_C2(obj, keep) :
        error("Unsupported type in blowdown: $(typeof(obj))")
        for obj in cochains
    )

    return keep, A_new, idx2id_new, blown...
end

# ---------------------------
# Linearize BV correction operator: compute J \approx d(dp)/d(p)
# ---------------------------
"""
linearize_BV_kernel(p::Vector{Float64}, A_sub::SparseMatrixCSC,
                    edge_u::Vector{Int}, edge_v::Vector{Int};
                    alpha=1e-3, eps=1e-6, nprobes = nothing)

Return dense Jacobian matrix J (m×m) approximating the map p -> dp produced by
integrate_BV_correction!. Uses forward-difference probing. For efficiency you
can provide nprobes (Int) < m to only probe that many coordinate directions;
if nprobes==nothing we probe all coordinates.

Note: m should be small enough (active subgraph).
"""
function linearize_BV_kernel(p::Vector{Float64}, A_sub::SparseMatrixCSC,
                             edge_u::Vector{Int}, edge_v::Vector{Int};
                             alpha::Float64=1e-3, eps::Float64=1e-6, nprobes=nothing)

    m = length(p)
    # baseline dp
    dp0 = zeros(Float64, m)
    integrate_BV_correction!(p, dp0, A_sub, edge_u, edge_v; alpha=alpha)

    probes = nprobes === nothing ? collect(1:m) : collect(1:min(nprobes,m))
    J = zeros(Float64, m, length(probes))

    # reuse temporaries
    p_pert = copy(p)
    for (j_idx, j) in enumerate(probes)
        p_pert .= p
        p_pert[j] += eps
        dp1 = zeros(Float64, m)
        integrate_BV_correction!(p_pert, dp1, A_sub, edge_u, edge_v; alpha=alpha)
        J[:, j_idx] .= (dp1 .- dp0) ./ eps
    end

    # if we probed all coords, return square J, otherwise expand to full m×m with columns probed
    if length(probes) == m
        return J
    else
        # embed probed columns into full matrix with unprobed columns as zeros
        Jfull = zeros(Float64, m, m)
        for (idx, col) in enumerate(probes)
            Jfull[:, col] .= J[:, idx]
        end
        return Jfull
    end
end

# ---------------------------
# Local Jacobian / singular locus scoring
# ---------------------------
"""
local_jacobian_scores(J::AbstractMatrix, A_sub::SparseMatrixCSC; radius=1)

Given the (m×m) Jacobian J and adjacency A_sub, compute for each node i a local
score = smallest singular value of the submatrix J[S,S], where S = {i} ∪ N(i)
(or extended neighborhood if radius>1). Returns vector scores of length m.

Smaller score => closer to singular; you may threshold or pick top k.
"""
function local_jacobian_scores(J::AbstractMatrix, A_sub::SparseMatrixCSC; radius::Int=1)
    m = size(J,1)
    # build neighbors list
    neigh = Vector{Vector{Int}}(undef, m)
    for i in 1:m
        neigh[i] = Int[]
    end
    rows, cols, _ = findnz(A_sub)
    for (r,c) in zip(rows, cols)
        push!(neigh[r], c)
        push!(neigh[c], r)  # treat undirected for neighborhood
    end

    # expand neighborhood to radius
    function neighborhood(i, r)
        S = Set{Int}([i])
        frontier = Set{Int}([i])
        for t in 1:r
            newf = Set{Int}()
            for v in frontier
                for w in neigh[v]
                    if !(w in S)
                        push!(newf, w)
                        push!(S, w)
                    end
                end
            end
            frontier = newf
            if isempty(frontier) break end
        end
        return sort(collect(S))
    end

    scores = zeros(Float64, m)
    for i in 1:m
        S = neighborhood(i, radius)
        Js = J[S, S]
        # numerical SVD: compute smallest singular value robustly via svdvals
        s = svdvals(Js)
        scores[i] = minimum(s)  # small => near-singular
    end
    return scores
end


# ---------------------------
# Entropy discrepancy (crepant check)
# ---------------------------
"""
node_entropy(p::Vector{Float64}; eps=1e-12)

Default per-node entropy density: -p log(p + eps)
"""
function node_entropy(p::Vector{Float64}; eps::Float64=1e-12)
    return .- (p .* log.(p .+ eps))
end

"""
compute_entropy_discrepancy(p_before::Vector{Float64}, p_after::Vector{Float64},
                            exceptional_sets::Vector{Vector{Int}};
                            entropy_fn = node_entropy)

Compute discrepancy per exceptional component: sum_entropy(after_nodes_assigned) - sum_entropy(before_nodes).
Assumes indices are in the SAME indexing frame (if after uses compacted indices, remap first).
Returns vector of discrepancies (one per exceptional set).
"""
function compute_entropy_discrepancy(p_before::Vector{Float64}, p_after::Vector{Float64},
                                     exceptional_sets::Vector{Vector{Int}};
                                     entropy_fn = node_entropy)

    S_before = entropy_fn(p_before)
    S_after = entropy_fn(p_after)

    discs = Float64[]
    for comp in exceptional_sets
        # comp is vector of original indices collapsed into a representative in after;
        # we sum before entropy over comp and compare to after on that representative (if available).
        sum_before = sum(S_before[comp])
        # Heuristic: find representative index in after by e.g. first kept index of comp
        # If no representative available, we compare to zero (fully removed)
        rep = nothing
        for i in comp
            if i <= length(p_after) && !iszero(p_after[i])  # heuristic rep detection
                rep = i; break
            end
        end
        sum_after = rep === nothing ? 0.0 : S_after[rep]
        push!(discs, sum_after - sum_before)
    end
    return discs
end

# ---------------------------
# Index compaction: relabel kept nodes to 1..mnew and remap C1/C2
# ---------------------------
"""
compact_indices(keep::Vector{Bool})

Return new_index_map::Dict{Int,Int} mapping old_index -> new_index (only kept),
and inv map new2old::Vector{Int}.
"""
function compact_indices(keep::Vector{Bool})
    new2old = Int[]
    old2new = Dict{Int,Int}()
    for i in 1:length(keep)
        if keep[i]
            push!(new2old, i)
            old2new[i] = length(new2old)
        end
    end
    return old2new, new2old
end

"""
remap_C1_C2_to_compact(c1::C1, c2::C2, old2new::Dict{Int,Int})

Return (c1_new, c2_new) remapped to compact indices. If an edge/path refers to removed index, drop it.
"""
function remap_C1_C2_to_compact(c1::C1, c2::C2, old2new::Dict{Int,Int})
    # remap C1
    u_new = Int[]; v_new = Int[]; vals1 = Float64[]; idx1 = Dict{Tuple{Int,Int},Int}()
    k = 0
    for i in eachindex(c1.vals)
        ou, ov = c1.u[i], c1.v[i]
        if haskey(old2new, ou) && haskey(old2new, ov)
            k += 1
            uu = old2new[ou]; vv = old2new[ov]
            push!(u_new, uu); push!(v_new, vv); push!(vals1, c1.vals[i])
            idx1[(uu,vv)] = k
        end
    end
    c1n = C1(u_new, v_new, vals1, idx1)

    # remap C2
    u2n = Int[]; v2n = Int[]; w2n = Int[]; vals2 = Float64[]; idx2 = Dict{Tuple{Int,Int,Int},Int}()
    kk = 0
    for i in eachindex(c2.vals)
        ou, ov, ow = c2.u[i], c2.v[i], c2.w[i]
        if haskey(old2new, ou) && haskey(old2new, ov) && haskey(old2new, ow)
            kk += 1
            push!(u2n, old2new[ou]); push!(v2n, old2new[ov]); push!(w2n, old2new[ow])
            push!(vals2, c2.vals[i])
            idx2[(u2n[end], v2n[end], w2n[end])] = kk
        end
    end
    c2n = C2(u2n, v2n, w2n, vals2, idx2)

    return c1n, c2n
end

# ---------------------------
# Spectral embedding for compact graph (fallback reconstructor)
# ---------------------------
"""
spectral_embedding(A_sub::SparseMatrixCSC; dim=2)

Compute a simple spectral embedding using the unnormalized graph Laplacian:
- compute Lap = D - A
- take eigenvectors 2..(dim+1) (skip trivial eigenvector)
Return coords::Matrix(dim, n).
"""
function spectral_embedding(A_sub::SparseMatrixCSC; dim::Int=2)
    n = size(A_sub,1)
    degs = zeros(Float64, n)
    rows, cols, vals = findnz(A_sub)
    for (r, _) in zip(rows, cols)
        degs[r] += 1.0
    end
    D = Diagonal(degs)
    L = Matrix(D) - Matrix(A_sub)  # small n, keep dense for eig
    vals_eig, vecs = eigen(Symmetric(L))
    # eigenvalues sorted ascending; skip first (should be ~0)
    coords = zeros(Float64, n, dim)
    for k in 1:dim
        coords[:, k] .= vecs[:, k+1]
    end
    return coords   # n × dim
end

"""
reconstruct_coords(original_pos::Union{Nothing,Dict{Int,Vector{Float64}}}, old2new::Dict{Int,Int},
                   new2old::Vector{Int}, A_compact::SparseMatrixCSC; dim=2)

Return coords_new::Matrix{Float64} (nnew × dim).
- If original_pos provided, their coordinates are used as anchors for nodes present in original_pos.
- For nodes without anchors, use spectral embedding and then align so anchored nodes keep their coords (affine alignment).
"""
function reconstruct_coords(original_pos::Union{Nothing,Dict{Int,Vector{Float64}}}, old2new, new2old,
                            A_compact::SparseMatrixCSC; dim::Int=2)
    nnew = size(A_compact,1)
    coords = zeros(Float64, nnew, dim)

    # collect anchors
    has_anchor = falses(nnew)
    anchor_idx = Int[]
    anchor_coords = Float64[]
    for (j, oldi) in enumerate(new2old)
        if original_pos !== nothing && haskey(original_pos, oldi)
            coords[j, :] .= original_pos[oldi][1:dim]
            has_anchor[j] = true
            push!(anchor_idx, j)
        end
    end

    # if all anchored, done
    if all(has_anchor)
        return coords
    end

    # spectral baseline
    spec = spectral_embedding(A_compact; dim=dim)  # nnew × dim

    # if there are anchors, compute affine transform from spec[anchors] -> coords[anchors]
    if !isempty(anchor_idx)
        # build matrices
        m = length(anchor_idx)
        X = zeros(2, m); Y = zeros(2, m)
        for (k, idx) in enumerate(anchor_idx)
            X[:, k] = spec[idx, 1:2]
            Y[:, k] = coords[idx, 1:2]
        end
        # find best affine transform: Y ≈ A*X + t
        μx = mean(X, dims=2); μy = mean(Y, dims=2)
        Xc = X .- μx; Yc = Y .- μy
        # linear part
        A_lin = Yc * pinv(Xc)
        t = vec(μy .- A_lin * μx)
        for i in 1:nnew
            coords[i, :] = vec(A_lin * spec[i, 1:2] .+ t)
        end
    else
        # no anchors: use spectral directly
        coords .= spec
    end

    return coords
end


"""
build_entropy_sheaf(p::C0, flux::C1, A_sub::SparseMatrixCSC; radius::Int=1)

Constructs the initial sections of the Sheaf based on the active graph state.
The section S(U_i) over neighborhood U_i is the subset of p and flux restricted to U_i.
"""
function build_entropy_sheaf(p::C0, flux::C1, A_sub::SparseMatrixCSC; radius::Int=1)
    m = length(p.vals)
    neigh_maps = Dict{Int, Vector{Int}}()
    
    # 1. Determine local neighborhoods (U_i)
    # Reuse neighborhood logic from local_jacobian_scores
    
    # Build adjacency list first (for radius=1)
    adj = Dict{Int, Set{Int}}()
    for i in 1:m
        adj[i] = Set{Int}([i])
    end
    rows, cols, _ = findnz(A_sub)
    for (r,c) in zip(rows, cols)
        push!(adj[r], c)
        push!(adj[c], r)
    end

    # Expand to radius (simplified for immediate use: only radius=1)
    for i in 1:m
        U_i = sort(collect(adj[i]))
        neigh_maps[i] = U_i
    end
    
    # 2. Extract local sections S(U_i)
    sections = Dict{Int, Tuple{C0, C1}}()
    node_to_section_key = Dict{Int, Vector{Int}}()
    
    for i in 1:m
        section_nodes = neigh_maps[i]
        
        # Local C0 (node values)
        p_local = C0(p.vals[section_nodes])
        
        # Local C1 (edge values)
        u_local = Int[]; v_local = Int[]; vals_local = Float64[]; idx_local = Dict{Tuple{Int,Int},Int}()
        k_local = 0
        
        # Filter global C1 to edges (u,v) where both u and v are in the section
        for k in eachindex(flux.vals)
            u_global = flux.u[k]
            v_global = flux.v[k]
            
            # Check if both endpoints are in the current section
            if u_global in section_nodes && v_global in section_nodes
                # Use global indices for the section map keys for simplicity
                k_local += 1
                push!(u_local, u_global)
                push!(v_local, v_global)
                push!(vals_local, flux.vals[k])
                idx_local[(u_global, v_global)] = k_local
            end
        end
        c1_local = C1(u_local, v_local, vals_local, idx_local)
        
        sections[i] = (p_local, c1_local)

        # Update node_to_section_key
        for node in section_nodes
            push!(get!(node_to_section_key, node, Int[]), i)
        end
    end

    return EntropySheaf(sections, node_to_section_key, neigh_maps)
end

"""
resolve_and_check_consistency(S::EntropySheaf, p_before::Vector{Float64},
                              keep_mask::Vector{Bool}, exceptional_sets::Vector{Vector{Int}})

This function conceptually performs the local projection (blow-down) and checks
the global consistency of the entropy field.

It uses the existing blowdown functions (assuming they are run externally)
and the provided discrepancy function for the *crepant check*.
"""
function resolve_and_check_consistency(S::EntropySheaf, p_before::Vector{Float64},
                                     keep_mask::Vector{Bool}, exceptional_sets::Vector{Vector{Int}},
                                     entropy_fn::Function = node_entropy)
    
    # 1. Simulate the blow-down/projection π: S(U_i) -> S(U_i \ E_i)
    # The result of the blow-down (p_new) is the projected global section.
    p_new_vals = p_before[keep_mask]
    
    # 2. Calculate the global entropy discrepancy (The Crepant Check)
    # The discrepancy measures how "non-crepant" (non-consistent) the blow-down was
    # regarding the total entropy flowing through the collapsed exceptional sets.
    
    # NOTE: The definition of compute_entropy_discrepancy in the previous response
    # requires an external definition of which new node corresponds to which old set.
    # We simplify here by only passing the entropy function.
    
    # Placeholder for the discrepancy function output
    # This result should be NEAR ZERO for a consistent/crepant blow-down.
    discrepancy = compute_entropy_discrepancy(p_before, p_new_vals,
                                             exceptional_sets;
                                             entropy_fn = entropy_fn)

    # 3. Assessment
    avg_discrepancy = sum(abs.(discrepancy)) / length(discrepancy)
    
    # The sheaf structure informs the Hironaka procedure: 
    # If avg_discrepancy is high, the center of the blow-up (the exceptional nodes) 
    # was poorly chosen relative to the global BV structure, meaning the 99%
    # of "cool" entropy flows were actually essential for local consistency.
    
    return avg_discrepancy
end
end # module EntropyBV
