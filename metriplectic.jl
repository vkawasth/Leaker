using SparseArrays
using LinearAlgebra
using .EntropyBV   # adjust module name / relative include if needed

# small utility: enforce zero-sum correction (conservative)
function zero_sum!(v::AbstractVector)
    s = sum(v)
    if abs(s) > 0
        @inbounds for i in eachindex(v)
            v[i] -= s / length(v)
        end
    end
    return v
end

# Poisson (BV) kick: produce node correction vector from flux on active subgraph
"""
bv_poisson_kick!(dp_active, p_active, A_sub; alpha=1e-3, use_self_bracket=true)
- dp_active: view into global dp (modified in-place)
- p_active: view into global p (read-only)
- A_sub: adjacency of active subgraph
- alpha: scaling of the Hamiltonian kick (small)
"""
function bv_poisson_kick!(dp_active::AbstractVector{<:Real},
                          p_active::AbstractVector{<:Real},
                          A_sub::SparseMatrixCSC{Float64,Int};
                          alpha::Float64=1e-3,
                          perturb_kicks::Int=8)
    # Build edge index & flux (using EntropyBV helpers)
    edge_u, edge_v, idxmap = EntropyBV.build_edge_index(A_sub)
    ne = length(edge_u)
    # build flux C1 on the active subgraph
    flux_vals = zeros(Float64, ne)
    @inbounds for k in 1:ne
        u, v = edge_u[k], edge_v[k]
        flux_vals[k] = A_sub[u,v] * (p_active[u] + p_active[v]) / 2
    end
    flux = EntropyBV.c1_from_edgevals(edge_u, edge_v, idxmap, flux_vals)

    # compute derived bracket from BV
    bracket = EntropyBV.derived_bracket_from_Delta_general(flux, flux, A_sub)

    # convert to node correction vector
    corr_vec = EntropyBV.c1_to_node_correction(bracket, size(A_sub,1); convention=:in_minus_out)

    # ensure conservative (sum zero) before applying
    zero_sum!(corr_vec)

    # apply small Hamiltonian kick into dp_active (in-place)
    @inbounds @simd for i in 1:length(dp_active)
        dp_active[i] += alpha * corr_vec[i]
    end
    return nothing
end

# Metric half-step: reuse entropy_flow_step! but with dt_half and only metric part.
# We'll call your existing entropy_flow_step! with dt_small; if you prefer operator-level,
# extract the metric term into a function and call directly.
# For safety create a wrapper that calls entropy_flow_step! with appropriate mobility.
function metric_step!(p::Vector{Float64}, π::Vector{Float64}, A::SparseMatrixCSC,
                      dp::Vector{Float64}; dt::Float64, mobility=:diag, update_fraction=0.0)
    # Use update_fraction=0.0 to avoid random partial update inside; we want controlled update.
    entropy_flow_step!(p, π, A, dp; dt=dt, mobility=mobility, update_fraction=update_fraction)
    return nothing
end

# Port: blowdown wrapper that compresses p, A, and optional cochains (C1, C2)
"""
apply_blowdown_port!(p, A; top_fraction=0.01, cochains=())
- p : global probability vector
- A : global adjacency (SparseMatrixCSC)
- cochains: tuple of cochain objects (C1, C2, ...) to be passed to perform_blowdown
Returns a tuple (p_new, A_new, keep_mask, blown_cochains...)
"""
function apply_blowdown_port!(p::Vector{Float64}, A::SparseMatrixCSC{Float64,Int};
                              top_fraction::Float64=0.01, cochains=())
    # call perform_blowdown on the active graph A (we treat A as the A_sub here)
    # cochains is a Tuple e.g. (C1, C2)
    keep, A_new, blown... = EntropyBV.perform_blowdown(p, A, cochains...; top_fraction=top_fraction)

    # compress p and renormalize
    p_new = p[keep]
    p_new .= max.(p_new, 1e-15)
    p_new ./= sum(p_new)

    return p_new, A_new, keep, blown
end

# Full metriplectic step (Strang splitting)
"""
metriplectic_step!(p, π, A, dp; dt, mobility, alpha_poisson, port_cfg)
performs: metric(dt/2) -> Poisson(dt) -> metric(dt/2) -> optional port
- port_cfg::Dict or nothing; if Dict with :every => k then apply port every k steps
"""
function metriplectic_step!(p::Vector{Float64}, π::Vector{Float64}, A::SparseMatrixCSC{Float64,Int}, dp::Vector{Float64};
                            dt::Float64=1e-3, mobility=:diag, alpha_poisson::Float64=2e-3,
                            port_cfg = nothing)
    # metric half-step
    fill!(dp, 0.0)
    metric_step!(p, π, A, dp; dt=dt/2, mobility=mobility, update_fraction=0.0)
    @inbounds @simd for i in eachindex(p) p[i] += (dt/2) * dp[i] end
    p .= max.(p, 1e-15); p ./= sum(p)

    # Poisson (BV) full-step: build active subgraph for the important nodes
    # choose top_fraction small (use local entropy) for speed
    local_ent = @. -p * log2(max(p, 1e-20))
    thresh = quantile(local_ent, 1 - 0.15)     # use top 15% as before
    active_global = findall(>=(thresh), local_ent)
    if length(active_global) >= 8
        _, _, A_sub = EntropyBV.build_active_subgraph(active_global, A)
        # views into active slices
        p_active = @views p[active_global]
        dp_active = zeros(length(active_global))
        # apply BV Poisson kick into dp_active
        bv_poisson_kick!(dp_active, p_active, A_sub; alpha=alpha_poisson)
        # write dp_active back into global dp (scatter add)
        @inbounds for (loc, glob) in enumerate(active_global)
            dp = dp_active[loc]
            p[glob] += dt * dp    # treat as explicit kick
        end
        p .= max.(p, 1e-15); p ./= sum(p)
    end

    # metric half-step
    fill!(dp, 0.0)
    metric_step!(p, π, A, dp; dt=dt/2, mobility=mobility, update_fraction=0.0)
    @inbounds @simd for i in eachindex(p) p[i] += (dt/2) * dp[i] end
    p .= max.(p, 1e-15); p ./= sum(p)

    # port (optional) — port_cfg example: Dict(:every=>50, :top_fraction=>0.01)
    if port_cfg !== nothing
        every = get(port_cfg, :every, 0)
        stepno = get(port_cfg, :stepno, 0)
        if every > 0 && (stepno % every == 0)
            # perform blowdown on full p/A
            cochs = get(port_cfg, :cochains, ())
            p_new, A_new, keep, blown = apply_blowdown_port!(p, A; top_fraction=get(port_cfg,:top_fraction,0.01), cochains=cochs)
            return (p_new, A_new, keep, blown)
        end
    end

    return (p, A, nothing, nothing)
end
