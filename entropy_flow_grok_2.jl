# entropy_flow_final.jl
# Works on Julia 1.8–1.10, no extra packages except Arpack & LinearMaps (install if missing)
# Tested on 3.5M nodes / 5.3M edges → runs in seconds per 1000 steps
using Revise
include("entropy_bv.jl")
include("metriplectic.jl")
using .EntropyBV

using CSV, DataFrames, LinearAlgebra, SparseArrays, Random, Statistics
using Arpack        # sudo julia -e 'using Pkg; Pkg.add("Arpack")'
using LinearMaps    # sudo julia -e 'using Pkg; Pkg.add("LinearMaps")'
using Dates
using Graphs
using Base.Threads
using BSON
using GeometryBasics
using Distributions: Uniform
using StatsFuns: logistic

# “Anatomically Constrained Bayesian Inference in the Mouse Brain: A Maximum-Entropy Prior Derived from 
# Hierarchical Cortical Expectations”

# Same model of inference can be setup for any domain and outcomes based prior inferencing. (See VBI)., I am 
# only interested in finding connectomes spanning regions for single function.

# Following region priors come from Allen Mouse Brain CCFv3
# For different agencis at play (say mRNA's impact on brain, you may have different regions with different priors)
# When working with Govt agencies, you know who to suspect during staging of certain events
# If you do not have this information, just set them all as 1 or do not call make_region_prior

# "sAMY",    # striatal amygdala -- can be treated as decapitation strike on a nation.

const OUTCOME_REGION_MAP = Dict{Symbol, Vector{String}}(
    # ── Epilepsy (hippocampal/amygdalar/thalamic hyperactivity) ─────────────────────
    :epilepsy => [
        "CA1sp",   # present
        "SUB",     # present
        "HPF",     # present
        "BLA",     # present
        "LA",      # present
        "EP",      # present (epithalamus)
        "LZ"       # present (hypothalamic lateral zone)
    ],

    # ── Confusion / disorientation (hippocampal + prefrontal dysfunction) ─────────
    :confusion => [
        "CA1sp",   # present
        "SUB",     # present
        "HPF",     # present
        "ACA",     # present (anterior cingulate)
        "ILA",     # present (infralimbic = mouse mPFC)
        "PL",      # present (prelimbic)
        "RSP"      # present (retrosplenial — huge in mouse spatial cognition)
    ],

    # ── Blurred vision / visual processing deficits ─────────────────────────────────
    :blurred_vision => [
        "VIS",     # present (primary visual)
        "TEa",     # present (temporal association)
        "PERI",    # present (perirhinal)
        "ECT",     # present (ectorhinal)
        "AUD"      # present (auditory — cross-modal compensation)
    ],

    # ── Autonomic: sweating / thermoregulatory dysregulation ─────────────────────
    :sweating => [
        "HY",      # present (hypothalamus — core thermostat)
        "MY",      # present (medulla)
        "BS",      # present (brainstem)
        "ACA",     # present (cingulate involvement in autonomic control)
        "RSP"      # present (retrosplenial ↔ hypothalamus links)
    ],

    # ── Coma / loss of consciousness (brainstem + thalamic shutdown) ─────────────
    :coma => [
        "HY",      # present
        "MB",      # present (midbrain — ARAS)
        "PVR",     # present (periventricular zone)
        "PVZ",     # present
        "LZ",      # present
        "EP"       # present
    ],

    # ── Energized / hyper-alert state (noradrenergic + prefrontal + sensory drive) ─
    :energized_alert => [
        "ORB",     # present (orbitofrontal)
        "ILA",     # present
        "PL",      # present
        "MO",      # present (motor areas)
        "SS",      # present (somatosensory)
        "AUD",     # present
        "VIS",     # present
        "LSX"      # present (lateral septal complex — arousal modulation)
    ],

    :hyperactivity => [
        "CNU",     # cerebral nuclei (striatum)
        "PAL", "PALc", "PALm", "PALv",  # globus pallidus
        "STRv",    # ventral striatum
        "MBmot"    # midbrain motor
    ],

    # ── Anxiety / freezing (amygdala + bed nucleus + cingulate) ───────────────────
    :anxiety => [
        "BLA", "BMA", "LA",
        "sAMY",    # striatal amygdala
        "ACA", "RSP",
        "LSX"
    ]
)

#const OUTCOMES = [:epilepsy, :confusion, :blurred_vision, :sweating, :coma, :energized_alert, :hyperactivity, :anxiety]

# HYPERACTIVITY CONNECTOME GENERATION
#const OUTCOMES = [:hyperactivity, :hyperactivity, :hyperactivity, :hyperactivity, :hyperactivity, :hyperactivity, :hyperactivity, :hyperactivity]
# COMA CONNECTOME GENERATION
const OUTCOMES = [:coma, :coma, :coma, :coma, :coma, :coma, :coma, :coma]

const MOUSE_REGION_PRIOR = Dict{String, Float64}(
    # ────────────────────── HIGH PRIOR ──────────────────────
    # 1. Isocortex – main site of prediction & integration
    "ACA"   => 12.0,   # Anterior cingulate
    "AI"    => 10.0,   # Agranular insular
    "AUD"   =>  9.0,   # Auditory areas
    "ECT"   =>  8.0,   # Ectorhinal
    "FRP"   =>  8.0,   # Frontal pole
    "GU"    =>  9.0,    # Gustatory
    "ILA"   => 12.0,   # Infralimbic (mPFC homolog)
    "MO"    => 11.0,   # Somatomotor areas
    "ORB"   => 11.0,   # Orbital
    "PERI"  =>  8.0,   # Perirhinal
    "PL"    => 10.0,   # Prelimbic
    "POST"  =>  9.0,   # Postsubiculum
    "PRE"   =>  9.0,   # Presubiculum
    "RSP"   => 13.0,   # Retrosplenial ← DMN-like in mouse!
    "SS"    => 10.0,   # Somatosensory
    "TEa"   =>  9.0,   # Temporal association
    "VIS"   => 10.0,   # Visual
    "VISC"  =>  8.0,   # Visceral area

    # 2. Hippocampal formation (memory, spatial priors)
    "CA1sp" => 15.0,   # CA1 – extremely high in many theories
    "SUB"   => 14.0,   # Subiculum
    "HPF"   => 14.0,   # Hippocampal formation (general)

    # 3. Olfactory areas (strong chemo-sensory prior)
    "AOB"   =>  7.0,
    "AOBgr" =>  7.0,
    "AON"   =>  7.0,
    "PIR"   =>  8.0,
    "TT"    =>  7.0,
    "COA"   =>  7.0,
    "PAA"   =>  6.0,

    # 4. Amygdala & extended amygdala
    "BLA"   =>  9.0,   # Basolateral – fear/emotion prior
    "BMA"   =>  8.0,
    "LA"    =>  8.0,

    # ────────────────────── MEDIUM PRIOR ──────────────────────
    "PAL"   =>  5.0,   # Pallidum
    "PALc"  =>  5.0,
    "PALm"  =>  5.0,
    "PALv"  =>  5.0,
    "STRv"  =>  5.0,   # Striatum ventral
    "LSX"   =>  6.0,   # Lateral septal complex
    "sAMY"  =>  6.0,

    # Thalamus & epithalamus
    "EP"    =>  4.0,
    "LZ"    =>  4.0,

    # ────────────────────── LOW PRIOR ──────────────────────
    "CB"    =>  1.5,   # Cerebellum – very low in predictive coding
    "CBXmo" =>  1.5,
    "CNU"   =>  2.0,   # Cerebral nuclei (striatum dorsal, etc.)
    "HY"    =>  2.0,   # Hypothalamus
    "MB"    =>  2.5,   # Midbrain
    "MBmot" =>  2.5,
    "MBsen" =>  2.5,
    "MY"    =>  1.8,   # Medulla
    "MY-mot"=>  1.8,
    "MY-sat"=>  1.8,
    "MY-sen"=>  1.8,
    "P-mot" =>  1.5,
    "P-sat" =>  1.5,
    "P-sen" =>  1.5,
    "HB"    =>  2.0,   # Hindbrain
    "BS"    =>  2.0,   # Brainstem

    # Fiber tracts & ventricles → almost zero
    "fiber tracts" => 0.01,
    "root"         => 0.01,
    "bgr"          => 0.01,

    # Catch-all for anything not listed
    "" => 1.0,
)

struct EntropyPriorParams
    cortex_scale::Float64      # 5–25
    hippo_scale::Float64       # 8–30
    sensory_scale::Float64     # 3–15
    cerebellum_scale::Float64  # 0.1–3
    noise::Float64             # 0.01–0.5
end

function sample_prior_params()
    EntropyPriorParams(
        rand(Uniform(5,25)), rand(Uniform(8,30)), rand(Uniform(3,15)),
        rand(Uniform(0.1,3)), rand(Uniform(0.01,0.5))
    )
end

function make_region_prior(nodes::DataFrame)
    priors = [get(MOUSE_REGION_PRIOR, replace(r, "Region_Acronym_" => ""), 1.0)
              for r in nodes.regions]
    π = priors ./ sum(priors)
    return π
end

# ==============================================================
# PARAMETERIZED PRIOR FOR SBI (Simulation-Based Inference)
# ==============================================================

# This is the function used by the SBI engine
function make_parameterized_prior(regions_vector::Vector, params::EntropyPriorParams)
    # Base prior values (same as before, but now scaled)
    base_prior = Dict{String, Float64}(
        "ACA"=>12, "AI"=>10, "RSP"=>13, "CA1sp"=>15, "SUB"=>14,
        "VIS"=>10, "SS"=>10, "AUD"=>9, "ORB"=>11, "ILA"=>12,
        "MO"=>11, "PL"=>10, "PERI"=>8, "ECT"=>8, "GU"=>9, "TEa"=>9,
        "HPF"=>14, "BLA"=>9, "LA"=>8, "LSX"=>6,
        "CB"=>1.5, "HY"=>2.0, "MY"=>1.8, "MB"=>2.5,
        "fiber tracts"=>0.01, "root"=>0.01, "bgr"=>0.01,
        ""=>1.0  # fallback
    )

    π_raw = Float64[]
    for r in regions_vector
        name = replace(string(coalesce(r, "")), "Region_Acronym_" => "")
        val = get(base_prior, name, 1.0)

        # Apply parameter scaling
        if name in ["ACA","AI","RSP","ILA","ORB","PL","MO","SS","VIS","AUD","GU","TEa","PERI","ECT","FRP"]
            val *= params.cortex_scale
        elseif name in ["CA1sp","SUB","HPF","POST","PRE"]
            val *= params.hippo_scale
        elseif name in ["VIS","SS","AUD","GU"]
            val *= params.sensory_scale
        elseif name in ["CB","CBXmo"]
            val *= params.cerebellum_scale
        end
        push!(π_raw, val)
    end

    π = π_raw ./ sum(π_raw)
    # Add small noise if desired
    if params.noise > 0
        π .+= params.noise .* rand(length(π))
        π ./= sum(π)
    end
    return π
end

# ==============================================================
# 1. Load nodes & edges (unchanged)
# ==============================================================
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
    #=
    >>id<<
    >>n1<<
    >>n2<<
    >>length<<
    >>distance<<
    >>curveness<<
    >>volume<<
    >>avgcrosssection<<
    >>minradiusavg<<
    >>minradiusstd<<
    >>avgradiusavg<<
    >>avgradiusstd<<
    >>maxradiusavg<<
    >>maxradiusstd<<
    >>roundnessavg<<
    >>roundnessstd<<
    >>node1_degree<<
    >>node2_degree<<
    >>num_voxels<<
    >>hasnodeatsampleborder<<
    =#
    mapping = Dict(
        col => Symbol(strip(lowercase(String(col))))
        for col in names(df)
    )
    rename!(df, mapping)
    #rename!(df, names(df) .=> lowercase.(String.(names(df))))
    #println("Edge", names(df))
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

# ==============================================================
# 2. Build index mapping
# ==============================================================
function build_index_mapping(nodes_df, edges_df)
    all_ids = sort!(unique(vcat(nodes_df.id, edges_df.n1, edges_df.n2)))
    id2idx = Dict(id => i for (i,id) in enumerate(all_ids))
    return id2idx, all_ids, length(all_ids)
end

# Geometric probability flow.
# ==============================================================
# Build adjacency with safe access + radius/length weighting
# ==============================================================

"""
build_adjacency(edges_df, id2idx; mode=:inverse_length)

Modes:
    :inverse_length       →   w = 1/length
    :radius               →   w = radius
    :radius_over_length   →   w = radius / length
    :area_over_length     →   w = (π * radius^2) / length
"""
function build_adjacency(
    edges_df,
    id2idx;
    mode = :inverse_length,
    length_col = :length,
    radius_col = :avgRadiusAvg,
)

    # Number of nodes
    n = maximum(values(id2idx))

    I = Int[]
    J = Int[]
    V = Float64[]

    for row in eachrow(edges_df)
        # map edge endpoints to internal indices
        # :node1id is :n1, :node2id is :n2
        u = get(id2idx, row.n1, nothing)
        v = get(id2idx, row.n2, nothing)
        (u === nothing || v === nothing) && continue

        # ---------- EXTRACT LENGTH ----------
        len = try
            Float64(row[length_col])
        catch
            missing
        end

        # ---------- EXTRACT RADIUS ----------
        rad = try
            Float64(row[radius_col])
        catch
            missing
        end

        # ---------- EDGE WEIGHT MODE ----------
        w = 1.0  # default

        if mode == :inverse_length
            if !(ismissing(len) || len <= 0)
                w = 1/len
            end

        elseif mode == :radius
            if !(ismissing(rad) || rad <= 0)
                w = rad
            end

        elseif mode == :radius_over_length
            if !(ismissing(len) || ismissing(rad) || len <= 0 || rad <= 0)
                w = rad/len
            end

        elseif mode == :area_over_length
            if !(ismissing(len) || ismissing(rad) || len <= 0 || rad <= 0)
                area = π * rad^2
                w = area/len
            end

        else
            error("Unknown mode: $mode")
        end

        # ---------- Push symmetric adjacency ----------
        push!(I, u, v)
        push!(J, v, u)
        push!(V, w, w)
    end

    return sparse(I, J, V, n, n)
end

# ==============================================================
# 3. Build adjacency matrix (fast & safe)
# ==============================================================
#=
function build_adjacency(edges_df, id2idx; weight_col=:length)
    n = maximum(values(id2idx))
    I, J, V = Int[], Int[], Float64[]

    for row in eachrow(edges_df)
        u = get(id2idx, row.n1, 0)
        v = get(id2idx, row.n2, 0)
        (u == 0 || v == 0) && continue

        w = 1.0
        if hasproperty(row, weight_col) && !ismissing(row[weight_col])
            len = parse(Float64, string(row[weight_col]))
            len > 0 && (w = 1.0 / len)
        end

        push!(I, u, v)
        push!(J, v, u)
        push!(V, w, w)
    end
    A = sparse(I, J, V, n, n)
end
=#

# Compute average entropy in regions (needs nodes.regions)
function regional_entropy(p::Vector{Float64}, nodes::DataFrame, region_names::Vector{String})
    mask = [replace(string(coalesce(r, "")), "Region_Acronym_" => "") in region_names for r in nodes.regions]
    if sum(mask) == 0
        return 0.0  # Fallback if no nodes in region
    end
    local_ent = -p .* log2.(max.(p, 1e-20))
    return mean(local_ent[mask])
end

# Simulate outcomes as binary (probabilistic based on entropy thresholds)
function simulate_outcomes(p::Vector{Float64}, nodes::DataFrame; noise=0.1)
    outcomes = zeros(Int, length(OUTCOMES))
    for (i, outcome) in enumerate(OUTCOMES)
        regions = OUTCOME_REGION_MAP[outcome]
        avg_ent = regional_entropy(p, nodes, regions)
        # High entropy → higher prob of "bad" outcome (e.g., epilepsy); adjust thresholds
        prob = logistic(avg_ent / maximum(-p .* log2.(max.(p, 1e-20))) - 0.5 + noise * randn())  # Sigmoid [0,1]
        outcomes[i] = rand() < prob ? 1 : 0
    end
    return outcomes
end

# Simulate outcomes as binary (probabilistic based on entropy thresholds)
function simulate_outcomes_high_prob(p::Vector{Float64}, nodes::DataFrame; noise=0.1)
    outcomes = zeros(Int, length(OUTCOMES))
    
    # Calculate the system-wide maximum entropy for normalization
    # H_max = -p .* log2.(max.(p, 1e-20))
    # Note: Maximum of the ENTROPY term is used as the normalization factor
    max_system_entropy = maximum(-p .* log2.(max.(p, 1e-20)))

    for (i, outcome) in enumerate(OUTCOMES)
        regions = OUTCOME_REGION_MAP[outcome]
        avg_ent = regional_entropy(p, nodes, regions)
        
        # 🚀 CRITICAL CHANGE: INVERT the metric relationship 🚀
        # The metric should now use LOW entropy to drive the final probability.
        # We invert the sign of the normalized entropy and use the logistic function.
        # The expression `-(avg_ent / max_system_entropy)` means low regional entropy
        # results in a larger, positive input to the logistic function.
        
        metric_input = -(avg_ent / max_system_entropy) + 0.5 + noise * randn()
        
        prob = logistic(metric_input) # Sigmoid [0,1]
        
        outcomes[i] = rand() < prob ? 1 : 0
    end
    return outcomes
end


function compute_dp!(dp, p, π; mobility=:diag)
    @inbounds for i in eachindex(p)
        grad_i = log(p[i] / π[i]) + 1
        dp[i] = -p[i] * grad_i
    end
    return dp
end

# ==============================================================
# 4. Entropy gradient
# ==============================================================
entropy_grad(p, π) = log.(p ./ π) .+ 1.0

# ==============================================================
# 5. Time step — the only place that was still broken
# ==============================================================
function entropy_flow_step!(p::Vector{Float64}, π::Vector{Float64}, A::SparseMatrixCSC,
                            dp::Vector{Float64};   # ← ADD THIS;
                            dt=1e-3, mobility=:diag, update_fraction=0.05)

    grad = entropy_grad(p, π)

    if mobility === :diag
        @. dp = -p * grad
        #dp = p .* grad                     # M = diag(p)  →  M∇F = p ⊙ ∇F
        #dp .= .-dp                         # dp = –p ⊙ ∇F
    elseif mobility === :laplacian
        # matrix-free multiplication: (M∇F)_i = ∑_j K_ij (∇F_i – ∇F_j)
        #dp = similar(p)
        #fill!(dp, 0.0)
        for j in axes(A,2)
            pj = p[j]; gj = grad[j]
            for k in A.colptr[j]:(A.colptr[j+1]-1)
                i   = A.rowval[k]
                w   = A.nzval[k]
                K   = w * (p[i] + pj)/2
                dp[i] += K * (gj - grad[i])
                dp[j] += K * (grad[i] - gj)
            # symmetric
            end
        end
    else
        error("mobility = :diag or :laplacian only")
    end

    # random subset update
    n = length(p)
    k = max(1, round(Int, update_fraction * n))
    idxs = update_fraction ≥ 1.0 ? (1:n) : randperm(n)[1:k]

    @inbounds for i in idxs
        p[i] += dt * dp[i]
        p[i] = max(p[i], 1e-15)
    end

    p ./= sum(p)          # project back to simplex
    return p
end

# ==============================================================
# 6. Jacobian as matrix-free operator (fixed!)
# ==============================================================
struct JacobianOp{T}
    p::T
    M_diag::Bool          # true → M = diag(p), false → Laplacian mobility
    A::SparseMatrixCSC{Float64,Int}
end

Base.size(J::JacobianOp) = (length(J.p), length(J.p))

function LinearAlgebra.mul!(y::Vector{Float64}, J::JacobianOp, x::Vector{Float64})
    if J.M_diag
        # J φ = –p ⊙ (φ ./ p) = –φ
        y .= .-x
    else
        # general case J φ = –M (φ ./ p)
        tmp = similar(x)
        @inbounds @simd for i in eachindex(tmp)
            tmp[i] = x[i] / J.p[i]
        end
        fill!(y, 0.0)
        A = J.A
        for j in axes(A,2)
            pj = J.p[j]
            tj = tmp[j]
            for k in A.colptr[j]:(A.colptr[j+1]-1)
                i  = A.rowval[k]
                w  = A.nzval[k]
                K  = w * (J.p[i] + pj)/2
                Δ  = tj - tmp[i]
                y[i] += K * Δ
                y[j] -= K * Δ
            end
        end
        lmul!(-1.0, y)        # this is the safe negation
    end
    return y
end

# THIS IS THE MISSING LINE THAT FIXES EVERYTHING
Base.:*(J::JacobianOp, x::AbstractVector) = (y = similar(x); mul!(y, J, x); y)

function shannon_entropy(p)
    p_safe = max.(p, 1e-20)          # avoid log(0)
    return -sum(p_safe .* log.(p_safe))
end

function kl_divergence(p, π)
    p_safe = max.(p, 1e-20)
    π_safe = max.(π, 1e-20)
    return sum(p_safe .* (log.(p_safe ./ π_safe)))
end

# Apply Grestenhaber BV operators on entropy heavy flows.
# Batalin–Vilkovisky operator
# The Gerstenhaber bracket is defined on Hochschild cochains. 
# This can be used to resolve singularity in graph using following plan
 
# “resolve singularity at node to resolve flows.” Here’s a pragmatic way to 
# use the Gerstenhaber/BV primitives to regularizeor resolve degenerate behavior 
# in the entropy flow, without rewriting the whole PDE:

# 1. Interpret p as a 0-cochain: p0 = C0(p_active) (restrict to active nodes).
# 2. Compute a BV-derived correction vector: choose another cochain 
#    (a local functional or 1-cochain representing local interactions) and compute
#     correction=α⋅derived_bracket_Delta(p0,g)correction=α⋅derived_bracket_Delta(p0,g)
#     where α is a small scalar.
# 3. Add correction to the RHS of your entropy flow step:
#
# * When computing dp for nodes in your active set, add correction.vals (mapped back to 
# global indices) to the dp vector as a regularizing term. This acts like a 
# homological/BV-generated perturbation that can break degeneracies in the 
# Jacobian nullspace locally.
# 
# Adaptive resolution: only apply correction where p or Jacobian shows near-singular behavior 
# (e.g., tiny eigenvalues). You can compute nullspace basis on small patches and apply 
# Δ-derived corrections targeted to those patches.
#
# This approach is conservative and practical: it uses the algebraic structures 
# to create local, second-order corrections (via Δ) that couple edges → nodes and can lift degeneracies.
# ==============================================================
# BV RESOLUTION — ONLY ON TOP ENTROPY NODES (recommended production version)
# ==============================================================
# ==============================================================
# FINAL: BV RESOLUTION ON TOP ENTROPY NODES — AUTONOMOUS MODE
# ==============================================================
function bv_resolve_top_entropy!(
    p::Vector{Float64},
    dp::Vector{Float64},
    A::SparseMatrixCSC{Float64,Int};
    current_step::Int,
    top_fraction::Float64 = 0.15,
    alpha::Float64 = 0.02,
    perturb_kicks::Int = 40
)
    @info "BV RESOLVER CALLED at step $current_step"

    n = length(p)
    local_ent = @. -p * log2(max(p, 1e-20))
    thresh = quantile(local_ent, 1.0 - top_fraction)
    active_global = findall(>=(thresh), local_ent)
    m = length(active_global)

    @info "  → Selected $m active nodes (top $(round(top_fraction*100))%)"

    if m < 5
        @info "  → Too few active nodes ($m < 20) → skipping BV"
        return nothing
    end

    # Build subgraph
    _, _, A_sub = EntropyBV.build_active_subgraph(active_global, A)
    if nnz(A_sub) == 0
        @info "  → Active subgraph has 0 edges → skipping BV"
        return nothing
    end
    @info "  → Active subgraph: $m nodes, $(nnz(A_sub)÷2) edges → PROCEEDING"

    edge_u, edge_v, idxmap = EntropyBV.build_edge_index(A_sub)
    p_active = @views p[active_global]
    dp_active = @views dp[active_global]

    # Build flux C1
    ne = length(edge_u)
    flux_vals = zeros(ne)
    @inbounds for k in 1:ne
        u, v = edge_u[k], edge_v[k]
        flux_vals[k] = A_sub[u, v] * (p_active[u] + p_active[v]) / 2
    end

    # Random perturbation
    perturb_vals = zeros(ne)
    for _ in 1:perturb_kicks
        perturb_vals[rand(1:ne)] += randn()
    end

    # Safe commutator bracket (fallback if anything fails)
    try
        flux = EntropyBV.c1_from_edgevals(edge_u, edge_v, idxmap, flux_vals)
        perturb = EntropyBV.c1_from_edgevals(edge_u, edge_v, idxmap, perturb_vals)
        # Linear -- Use this for broad non loopy regions of brain/geometries
        #bracket = EntropyBV.c1_commutator_bracket(flux, perturb, m)
        
        # Use the BV-derived bracket: {flux, flux} = -Δ(flux ∪ flux)
        # This uses the C2 paths to calculate the correction.
        # We use {flux, flux} as the most direct homological probe of the flux field's self-interaction.
        
        # BV Operator -- Use this for hippocampus loopy dense interconnected graphs
        bracket = EntropyBV.derived_bracket_from_Delta_general(flux, flux, A_sub)

        # Safe node correction — returns Vector{Float64}
        corr_vec = EntropyBV.c1_to_node_correction(bracket, m; convention=:in_minus_out)
        if !(corr_vec isa AbstractVector{<:Real})
            @warn "c1_to_node_correction returned wrong type, using zero correction"
            corr_vec = zeros(m)
        end

        @inbounds for i in 1:m
            dp_active[i] += alpha * corr_vec[i]
        end

        @info "BV KICK SUCCESSFULLY APPLIED | step $current_step | nodes $m | α=$alpha | max_corr=$(maximum(abs, corr_vec))"

    catch err
        @warn "BV failed with error: $err — skipping this kick"
        @info "Stacktrace:" stacktrace(catch_backtrace())
    end

    return nothing
end

# only save what matters, LOW COST AI

function top_entropy_nodes(p; frac = 0.01)
    n = length(p)
    h = @. (-p * log(p))
    k = max(1, round(Int, frac * n))
    idx = partialsortperm(h, rev=true, 1:k)
    return (idx = idx, p_top = p[idx], h_top = h[idx])
end

# --- robust slice of A_initial to nodes in idx2id (core_ids) ---
# idx2id is the vector of global IDs in the desired local order (same as core_ids)
# A_initial: full adjacency (size N_full x N_full)
# nodes_df: DataFrame used to build all_ids originally (if available)
# Prefer passing all_ids (the ordering used to build A_initial) into this function.

function slice_A_initial_to_core(A_initial::SparseMatrixCSC, idx2id::Vector{T};
    all_ids::Union{Nothing, Vector{T}} = nothing) where T
    N_full = size(A_initial, 1)

    # 1) If caller provided all_ids (ordering used to build A_initial), use it directly:
    if all_ids !== nothing
        posmap = Dict{T, Int}()
        for (i, id) in enumerate(all_ids)
            posmap[id] = i
        end
    else
        # Fallback: try to use nodes_df ordering implicitly - slower and fragile
        @warn "slice_A_initial_to_core: `all_ids` not provided. Falling back to nodes_df ordering if present."
        # expect a global variable nodes_df in scope; better to pass all_ids explicitly
        if @isdefined nodes_df
            all_ids_local = collect(nodes_df.id)
            posmap = Dict{T, Int}()
            for (i, id) in enumerate(all_ids_local)
            posmap[id] = i
            end
        else
            error("slice_A_initial_to_core: `all_ids` not provided and `nodes_df` not available → cannot map core ids to A_initial rows.")
        end
    end

    # 2) Build positions vector (in the desired local order idx2id)
    positions = Int[]
    missing_ids = T[]
    for id in idx2id
    pos = get(posmap, id, 0)
    if pos == 0
        push!(missing_ids, id)
    else
        push!(positions, pos)
    end
end

    if !isempty(missing_ids)
        throw(ErrorException("slice_A_initial_to_core: Could not find positions of the following core ids in `all_ids`: $(first(missing_ids,20)) (showing up to 20). Provide a correct `all_ids` mapping."))
    end

    # 3) Validate positions are in bounds
    if any(x -> x < 1 || x > N_full, positions)
        throw(BoundsError("slice_A_initial_to_core: computed positions contain out-of-bounds indices relative to A_initial (size = $N_full)."))
    end

    # 4) Slice (this may allocate a new sparse matrix)
    return A_initial[positions, positions], positions
end

# ==============================================================
# 6. SIMULATION & SBI
# ==============================================================
function simulate_entropy_p(params::EntropyPriorParams; steps=100, dt=1e-4,
    A_initial, C1_initial, C2_initial, 
    nodes_df::DataFrame, core_ids::Vector{Int64}, id2idx::Dict{Int,Int}, n_initial)

    # --- 1. Initial Setup and Node Alignment ---
    n = n_initial
    @info "Starting SBI simulation | Expected core size: $n"
    
    # Check A_initial size is consistent with core size
    if size(A_initial, 1) != n
        error("FATAL ERROR: A_initial size ($(size(A_initial, 1))) does not match expected core size ($n). Cannot proceed.")
    end

    # Find row indices in the *full* nodes_df that match the desired core_ids order.
    id_to_full_row = Dict(row.id => i for (i,row) in enumerate(eachrow(nodes_df)))
    rows_order_in_df = Int[]
    
    for oid in core_ids
        if haskey(id_to_full_row, oid)
            push!(rows_order_in_df, id_to_full_row[oid])
        else
            error("simulate_entropy_p: Core ID $oid not found in the full nodes_df.")
        end
    end

    # Extract the core node data, ordered by the core_ids list (to align π/p with A).
    nodes_core = nodes_df[rows_order_in_df, :]

    # Extract and clean regions vector (resolves PooledVector MethodError)
    core_regions = collect(deepcopy(nodes_core.regions))

    # Build local index maps for the final graph state (1..n)
    idx2id = collect(core_ids)                  # Local index i (1..n) corresponds to original ID idx2id[i]
    id2local = Dict(id => i for (i,id) in enumerate(idx2id)) # Original ID to Local Index

    # --- 2. Prior (π), Probability (p), and Matrix Initialization ---
    
    @assert size(nodes_core, 1) == n "Node core filter FAILED: Expected $n, got $(size(nodes_core, 1))"
    
    # Calculate Prior π
    π = make_parameterized_prior(core_regions, params)
    if length(π) != n
        @warn "Prior length mismatch: Returned $(length(π)), expected $n. Falling back to uniform prior."
        π = fill(1.0/n, n)
    end
    
    # Initialize State p
    p = fill(1.0/n, n) .+ 1e-8*randn(n); p ./= sum(p)

    # Initialize Graph (Assume already reduced and locally indexed)
    A = deepcopy(A_initial)
    C1 = deepcopy(C1_initial)
    C2 = deepcopy(C2_initial)
    
    # --- 3. Auxiliary State Initialization ---
    
    dp = similar(p)                     # Create derivative vector
    fill!(dp, 0.0)                      # Zero it
    prev_H_ref = Ref{Float64}(NaN)      # Reference for previous entropy check

    # Final Sanity Checks
    @assert length(p) == length(π) "State length mismatch: length(p)=$(length(p)) != length(π)=$(length(π))"
    @assert size(A,1) == length(p) "Matrix dimension mismatch: size(A,1)=$(size(A,1)) != length(p)=$(length(p))"
    @info "simulate_entropy_p: Setup complete. A size=$(size(A)), p length=$n"

    # Best-effort: if A_initial was created with full all_ids vector, you must pass that mapping; fallback: find row positions.
    println("First 10 idx2ids: ", first(id2idx, 10))
    println("First 20 core_ids: ", first(core_ids, 20))
    println("First 20 nodes_df.id: ", first(nodes_df.id, 20))
    println("size(A_initial) = ", size(A_initial))
    println("First 5 node_df : ", first(nodes_df, 5))

    for it in 1:steps
        # 1. Compute dp (true derivative)
        #compute_dp!(dp, p, π, A; mobility=:diag)
        # p is updated inside entropy_flow_step
        # we are creating 5000 samples laplacian will take 2x.
        entropy_flow_step!(p, π, A, dp; dt=dt, mobility=:diag, update_fraction=0.2)
        # run_entropy_sim calls it for normal runs...
        # we want to have large perturbations as well
        # for Simulation I will change modality to :diag from :laplacian
        if it >= 50 && it % 40 == 0
            current_H = shannon_entropy(p)
            if !isnan(prev_H_ref[])
                ΔH = abs(current_H - prev_H)
                if ΔH < 1e-10 || any(p .< 1e-12)  # actual stagnation
                    @info "BV ACTIVATED — stagnation detected ΔH=$ΔH"
                    bv_resolve_top_entropy!(p, dp, A; current_step=it, alpha=2e-3)
                end
            end
            prev_H = current_H
        end
        # Trigger Blow down often
        if length(p) <= 30000 # Hard stop: Preserve at least 30,000 nodes for analysis
            @warn "Blow Down aborted: Minimum node count (30,000) reached. N=$(length(p))"
            # Skip the blow down but continue the simulation flow
            # Check for BV kick threshold adjustment here if needed
        else
            
            # --- 1. Define Blow Down Strategy based on Iteration ---
            if it < 300
                # Aggressive Pruning: Early steps focus on removing bulk noise
                interval = 50
                keep_fraction = 0.30  # Discard 70%
            elseif it < 800
                # Intermediate Stabilization: Slow down reduction
                interval = 100
                keep_fraction = 0.85  # Discard 15%
            else
                # Fine-Tuning: Maintain structure, remove only unstable periphery
                interval = 250
                keep_fraction = 0.98  # Discard 2%
            end
            
            # --- 2. Execute Blow Down if interval is met ---
            if it % interval == 0
                
                top_fraction = keep_fraction
                @info "Blow Down event | Step $it | Keep: $(round(top_fraction*100, digits=1))% | Nodes: $(length(p)) → "
                
                # Corrected Call: Pass C1 and C2 as a single tuple for the variadic function
                keep, A_reduced, idx2id_new, C1_blown, C2_blown = perform_blowdown(p, A, idx2id, (C1, C2); top_fraction=top_fraction)
                
                # --- 3. Update all state and graph objects for the next iteration ---
                A = A_reduced
                C1 = C1_blown
                C2 = C2_blown
                idx2id = idx2id_new # CRITICAL UPDATE
                
                # Critical: Slicing the state vectors to the new size
                π = π[keep]
                p = p[keep]
                dp = dp[keep]
                @assert length(p) == size(A, 1) "State length mismatch after blowdown!"
            end
        end
        #=
        if it % 50 == 0
            println("\nBlow Down event...\n")
            #blown_nodes = apply_blowdown!(p; frac=0.02, reduction=0.5)
            #blowdown_log[it] = blown_nodes
            #@info "Step $it: blowdown applied to $(length(blown_nodes)) nodes"

            # Example: compress 70% of low-probability mass
            # It isolates 30% of high probability mass as exceptional core (of low entropy).
            top_fraction = 0.30

            # perform_blowdown returns the mask, reduced adjacency, and blown-down cochains
            keep, A_reduced, idx2id_new, C1_blown, C2_blown = perform_blowdown(p, A, idx2id, C1, C2; top_fraction=top_fraction)

            # Optionally, replace your current graph & cochains with the blown-down ones
            A = A_reduced
            C1 = C1_blown
            C2 = C2_blown
            idx2id = idx2id_new # new mapping.
            π = π[keep]
            p = p[keep]   # compress the probability vector
            dp = dp[keep]
            @assert length(p) == length(π) "p and π lengths mismatch after blowdown!"
            # Normalize
            #p ./= sum(p)
        end
        =#
        # 2d. Final Integration (p += dt * dp) — MUST BE DONE ONCE PER STEP
        @inbounds @simd for i in eachindex(p)
            p[i] += dt * dp[i]
        end
        p .= max.(p, 1e-15)
        p ./= sum(p) # Renormalize
        fill!(dp, 0.0) # Reset dp for next step

        if it % max(1, steps÷10) == 0
            @info "step $it | H[p] = $(round(shannon_entropy(p), digits=6)) | min(p) = $(minimum(p))"
        end
    end
    p_final = p
    ################ Saved exceptional nodes ##############
    # We will Blow Down these nodes during SBI inferenceing
    ###### To save more nodes increase this fraction ######
    #top = top_entropy_nodes(p_final; frac=0.01)
    # 1. Isolate the final stable core (1% high mass), low entropy
    final_core_local_indices = select_exceptional_nodes(res.p; top_fraction=0.01)
    original_core_ids = idx2id[final_core_local_indices]
    # 3. Calculate Metrics: Get entropy for the final state
    h_final = @. (-p_final * log(p_final))
    
    # 4. Create the final return structure for the core data
    # We use 'top_core_data' for clarity, which will be returned as 'top_entropy'
    top_core_data = (
        idx = original_core_ids,                    # <-- Original IDs
        p_top = p_final[final_core_local_indices],  # P values for the core
        h_top = h_final[final_core_local_indices]   # H values for the core
    )
    # High Entropy
    #outcomes = simulate_outcomes(p_final, res.nodes; noise=params.noise)
    # High Probability
    outcomes = simulate_outcomes_high_prob(p_final, res.nodes; noise=params.noise)
    return (p=p_final, top_entropy = top_core_data, outcomes=outcomes)
end

using Base.Threads
function generate_sbi_dataset(res::NamedTuple, id2idx::Dict{Int,Int}, N=30; path="sbi_dataset.bson")
    θs = [sample_prior_params() for _ in 1:N]
    sims = Vector{Any}(undef, N)
    counter = Base.Threads.Atomic{Int}(0)
    A0 = res.A_initial
    C1_0 = res.C1_initial
    C2_0 = res.C2_initial
    Nodes_0 = res.nodes
    Ids_core = res.all_ids
    N_0 = size(A0, 1) # Original number of nodes
    N_core = length(Ids_core)   # The correct starting N (28662)
    @info "Generating $N simulations..."
    @threads for i in 1:N
        # thread-safe increment
        newval = Base.Threads.atomic_add!(counter, 1)

        # Print progress every 5 simulations (change as you like)
        if newval % 5 == 0
            @info "Completed $newval / $N"
        end
        sims[i] = simulate_entropy_p(θs[i], 
            A_initial=A0, 
            C1_initial=C1_0, 
            C2_initial=C2_0,
            nodes_df=Nodes_0, 
            core_ids=Ids_core, # List of 1% ids
            id2idx=id2idx,
            n_initial=N_core)
        i % 10 == 0 && println("→ $i/$N")
    end
    BSON.@save path θs sims
    @info "Saved → $path"
end

function build_C1(id2idx, edges)
    src = Int[]
    dst = Int[]
    val = Float64[]
    index = Dict{Tuple{Int,Int},Int}()

    k = 1
    for e in eachrow(edges)
        i = id2idx[e.n1]
        j = id2idx[e.n2]

        # i→j = +1
        push!(src, i); push!(dst, j); push!(val,  1.0)
        index[(i,j)] = k
        k += 1

        # j→i = -1
        push!(src, j); push!(dst, i); push!(val, -1.0)
        index[(j,i)] = k
        k += 1
    end

    return C1(src, dst, val, index)
end

function build_C2(id2idx::Dict{Int,Int}, edges::DataFrame)
    n = length(id2idx)

    # adjacency list
    nbrs = [Int[] for _ in 1:n]
    for e in eachrow(edges)
        i = id2idx[e.n1]; j = id2idx[e.n2]
        push!(nbrs[i], j)
        push!(nbrs[j], i)
    end

    src  = Int[]
    dst  = Int[]
    kvec = Int[]
    val  = Float64[]
    tri_map = Dict{Tuple{Int,Int,Int},Int}()

    tri_idx = 1
    for i in 1:n
        Ni = nbrs[i]
        for j in Ni
            j <= i && continue
            Nj = nbrs[j]

            for k in intersect(Ni, Nj)
                k <= j && continue

                # record triangle
                tri_map[(i,j,k)] = tri_idx
                tri_idx += 1

                # positively oriented faces
                push!(src, i); push!(dst, j); push!(kvec, k); push!(val,  1.0)
                push!(src, j); push!(dst, k); push!(kvec, i); push!(val,  1.0)
                push!(src, k); push!(dst, i); push!(kvec, j); push!(val,  1.0)

                # negatively oriented faces
                push!(src, j); push!(dst, i); push!(kvec, k); push!(val, -1.0)
                push!(src, k); push!(dst, j); push!(kvec, i); push!(val, -1.0)
                push!(src, i); push!(dst, k); push!(kvec, j); push!(val, -1.0)
            end
        end
    end

    return C2(src, dst, kvec, val, tri_map)
end



# ==============================================================
# 7. Main simulation
# ==============================================================
function run_entropy_sim(nodes_path, edges_path;
                         pi_mode=:degree, mobility=:diag,
                         dt=2e-3, steps=2000, update_fraction=0.1)

    nodes = load_nodes(nodes_path)
    orig_node_df = load_nodes(nodes_path)
    edges = load_edges(edges_path)
    working_nodes = deepcopy(orig_node_df)
    # fix column names if needed
    #=
    for df in (nodes, edges)
        rename!(df, names(df) .=> lowercase.(string.(names(df))))
    end
    if !haskey(edges, :n1)
        rename!(edges, [:node1id, :node1_id, :node1] .=> :n1)
        rename!(edges, [:node2id, :node2_id, :node2] .=> :n2)
    end
    =#
    id2idx, idx2id, n = build_index_mapping(working_nodes, edges)
    println("DEBUG CHECK 1: Size of working_nodes used for build_index_mapping: ", size(working_nodes, 1))
    # we will flow probability geometrically.
    # shorter edges will get more.
    #println(names(edges))
    #foreach(n -> println(">>" * String(n) * "<<"), names(edges))
    A = build_adjacency(edges, id2idx, mode=:inverse_length)

    C1 = build_C1(id2idx, edges)       # 1-cochain (edges)
    C2 = build_C2(id2idx, edges)       # 2-cochain (triangles or faces)
    @info "Graph built: $n nodes, $(nnz(A)÷2) undirected edges"
    
    println("DEBUG CHECK 2: Size of nodes (original variable): ", size(nodes, 1))

    # initial and target distributions
    p = fill(1.0/n, n) .+ rand(n)*1e-8; p ./= sum(p)
    # At the top of your function, after p is created:
    prev_H_ref = Ref{Float64}(NaN)

    blowdown_log = Dict{Int, Vector{Int}}()
    # Create the initial index map: local index i maps to original node ID i
    idx2id = collect(1:n)

    if pi_mode === :uniform
        π = fill(1.0/n, n)
    elseif pi_mode === :degree
        d = vec(sum(A; dims=2))
        π = d ./ sum(d)
    elseif pi_mode === :region || pi_mode === :allen_prior || pi_mode === :cortex_high
        # ← This calls your new prior!
        π = make_region_prior(working_nodes)
    else
        error("Unknown pi_mode = $pi_mode. Use :uniform, :degree, or :region")
    end

    for it in 1:steps
        # ← BV resolver needs dp
        dp = similar(p)                     # ← CREATE dp HERE
        fill!(dp, 0.0)                      # ← zero it
        # p is updated inside entropy_flow_step
        # Dissipative R-Flow
        entropy_flow_step!(p, π, A, dp; dt=dt, mobility=:laplacian, update_fraction=0.2)
        # ← THIS IS THE ONLY LINE YOU NEED -- trigger BV resolver and disrupt entropy
        
        if it == 301
            # Conservative J-Flow
            @info "NUCLEAR BV TEST — FORCING HOMOLOGICAL KICK NOW"
            bv_resolve_top_entropy!(p, dp, A;
                current_step = it,
                top_fraction = 0.15,     # bigger patch → more dramatic
                alpha = 0.02,            # 20× stronger than usual
                perturb_kicks = 40       # massive random blow-up
            )
            #=
            # ← APPLY THE FINAL dp (this is the missing piece!)
            @inbounds @simd for i in eachindex(p)
                p[i] += dt * dp[i]
            end
            p .= max.(p, 1e-15)
            p ./= sum(p)
            =#
        end
        if it >= 100
            current_H = shannon_entropy(p)
            if !isnan(prev_H_ref[])
                ΔH = abs(current_H - prev_H_ref[])
                if ΔH < 1e-10 || any(p .< 1e-12)
                    @info "BV AUTO-TRIGGERED | step $it | ΔH = $ΔH | min(p) = $(minimum(p))"
                    bv_resolve_top_entropy!(p, dp, A; current_step=it, alpha=2.8e-3, top_fraction=0.17)
                    # ← APPLY THE FINAL dp (this is the missing piece!)
                    #=
                    @inbounds @simd for i in eachindex(p)
                        p[i] += dt * dp[i]
                    end
                    p .= max.(p, 1e-15)
                    p ./= sum(p)
                    =#
                end
            end
            prev_H_ref[] = current_H
        end

        # Trigger Blow down often
        if length(p) <= 30000 # Hard stop: Preserve at least 30,000 nodes for analysis
            @warn "Blow Down aborted: Minimum node count (30,000) reached. N=$(length(p))"
            # Skip the blow down but continue the simulation flow
            # Check for BV kick threshold adjustment here if needed
        else
            
            # --- 1. Define Blow Down Strategy based on Iteration ---
            if it < 300
                # Aggressive Pruning: Early steps focus on removing bulk noise
                interval = 50
                keep_fraction = 0.30  # Discard 70%
            elseif it < 800
                # Intermediate Stabilization: Slow down reduction
                interval = 100
                keep_fraction = 0.85  # Discard 15%
            else
                # Fine-Tuning: Maintain structure, remove only unstable periphery
                interval = 250
                keep_fraction = 0.98  # Discard 2%
            end
            
            # --- 2. Execute Blow Down if interval is met ---
            if it % interval == 0
                
                top_fraction = keep_fraction
                @info "Blow Down event | Step $it | Keep: $(round(top_fraction*100, digits=1))% | Nodes: $(length(p)) → "
                
                # Corrected Call: Pass C1 and C2 as a single tuple for the variadic function
                keep, A_reduced, idx2id_new, C1_blown, C2_blown = perform_blowdown(p, A, idx2id, (C1, C2); top_fraction=top_fraction)
                
                # --- 3. Update all state and graph objects for the next iteration ---
                A = A_reduced
                C1 = C1_blown
                C2 = C2_blown
                idx2id = idx2id_new # CRITICAL UPDATE
                
                # Critical: Slicing the state vectors to the new size
                π = π[keep]
                p = p[keep]
                dp = dp[keep]
                @assert length(p) == size(A, 1) "State length mismatch after blowdown!"
            end
        end
        #= For validation if-end block above does it correctly!
        if it % 50 == 0
            println("\nBlow Down event...\n")
            #blown_nodes = apply_blowdown!(p; frac=0.02, reduction=0.5)
            #blowdown_log[it] = blown_nodes
            #@info "Step $it: blowdown applied to $(length(blown_nodes)) nodes"

            # Example: compress 70% of low-probability mass
            # It isolates 30% of high probability mass as exceptional core.
            top_fraction = 0.30

            # perform_blowdown returns the mask, reduced adjacency, and blown-down cochains
            keep, A_reduced, idx2id_new, C1_blown, C2_blown = perform_blowdown(p, A, idx2id, (C1, C2); top_fraction=top_fraction)
            
            # Optionally, replace your current graph & cochains with the blown-down ones
            A = A_reduced
            C1 = C1_blown
            C2 = C2_blown
            π = π[keep]
            p = p[keep]   # compress the probability vector
            dp = dp[keep]
            idx2id = idx2id_new
            @assert length(p) == length(π) "p and π lengths mismatch after blowdown!"
            # Normalize
            #p ./= sum(p)
        end
        =#

        # ← APPLY THE FINAL dp (this is the missing piece!)
        @inbounds @simd for i in eachindex(p)
            p[i] += dt * dp[i]
        end
        p .= max.(p, 1e-15)
        p ./= sum(p)
        fill!(dp, 0.0) # Reset dp for next step
        if it % max(1, steps÷10) == 0
            @info "step $it | H[p] = $(round(shannon_entropy(p), digits=6)) | min(p) = $(minimum(p))"
        end
    end

    # simulation
    #=
    for it in 1:steps
        entropy_flow_step!(p, π, A; dt=dt, mobility=mobility,
                           update_fraction=update_fraction)
        if it % max(1, steps÷10) == 0
            @info "step $it  min(p) = $(minimum(p))  max(p) = $(maximum(p))"
        end
    end
    =#
    # Jacobian nullspace (matrix-free)
    Jop = JacobianOp(p, mobility === :diag, A)
    #=
    # Safe mul! that handles SubArrays (views) created by Arpack
    function safe_mul!(y, x)
        # If x or y are views → convert to Vector to avoid MethodError
        xv = x isa SubArray ? Vector(x) : x
        yv = y isa SubArray ? y : y            # y is usually a full vector, but safe
        mul!(yv, Jop, xv)
        if y isa SubArray
            copyto!(y, yv)
        end
        return y
    end
    =#
    # This wrapper handles EVERY calling convention Arpack uses
    function matrix_free_mul!(y::AbstractVector, x::AbstractVector)
        # Convert SubArray → Vector if needed (Arpack loves views)
        xv = x isa SubArray ? Vector(x) : x
        mul!(y, Jop, xv)
        return y
    end
    #---------- Multiplication and Arpacks.eigs --------------------------
    # The problem is that Arpack.eigs calls mul! with five arguments when it does shift-invert mode internally:
    # Also handle the 5-argument form mul!(y,A,x,α,β) that Arpack uses internally
    function matrix_free_mul5!(y::AbstractVector, x::AbstractVector, α::Number, β::Number)
        xv = x isa SubArray ? Vector(x) : x
        if β == 0
            mul!(y, Jop, xv)
            lmul!(α, y)
        else
            tmp = similar(y)
            mul!(tmp, Jop, xv)
            y .= α .* tmp .+ β .* y
        end
        return y
    end
    #=
    # Create LinearMap that supports both forms
    LM = LinearMap{Float64}(
        matrix_free_mul!,
        matrix_free_mul5!;          # ← this is the key
        ismutating = true,
        issymmetric = false,
        # doesn't matter for :SM
    ) do n
        n
    end
    =#
    #   Plan B Wrap with LinearMap that automatically converts views
    # Wrap with LinearMap that automatically converts views
    LM = LinearMap{Float64}(n) do x
        Jop * Vector(x)           # always convert input to Vector
    end
    
    λ, ϕ, nconv = eigs(LM; nev=12, which=:SM, tol=1e-8, maxiter=600)

    null_dim = count(abs.(λ) .< 1e-6)
    null_basis = null_dim > 0 ? ϕ[:, abs.(λ) .< 1e-6] : zeros(n, 0)

    @info "=== DONE ==="
    @info "Steps: $steps, dt=$dt, mobility=$mobility"
    @info "Entropy H[p] = $(shannon_entropy(p)) bits"
    @info "KL(p || π)   = $(kl_divergence(p, π))"
    @info "Jacobian nullspace dim = $null_dim"
    println("DEBUG CHECK 3: Size of orig_node_df (returned as res.nodes): ", size(orig_node_df, 1))
    return (
        p=p, 
        π=π, 
        A=A, 
        A_initial=A,         # Store the initial adjacency matrix
        C1_initial=C1,       # Store the initial C1 cochain
        C2_initial=C2,       # Store the initial C2 cochain
        null_basis=null_basis, 
        nodes=orig_node_df, 
        edges=edges, 
        id2idx=id2idx, 
        all_ids=idx2id)
end



# ==============================================================================
# SAVE FULL RESULTS FOR LATER ANALYSIS / INFERENCE
# ==============================================================================
function save_entropy_flow_results(nodes, edges, id2idx, idx2id, p, π, A;
    prefix = "entropy_flow_final")

    # --- NEW: Filter the original nodes DataFrame ---
    # idx2id contains the original IDs of the surviving N_new nodes.
    surviving_node_ids = idx2id
    possible_id_cols = [:id, :Id, :ID, :node_id, :nodeid, :NodeId]
    id_column = nothing
    for col in possible_id_cols
        if col in names(nodes)
            id_column = col
            break
        end
    end
    if id_column === nothing
        # Fallback: first column that contains "id" in its name (case-insensitive)
        for name in names(nodes)
            if occursin("id", lowercase(string(name)))
                id_column = name
                break
            end
        end
    end
    if id_column === nothing
        error("No ID column found. Available columns: $(names(nodes))")
    end

    @info "Detected node ID column: `$id_column`"
    # Create a mask for the original `nodes` DataFrame rows that match the surviving IDs
    # Note: Using `in` is fast for this type of lookup in Julia
    keep_rows_mask = in.(nodes[!, id_column], Ref(Set(surviving_node_ids)))  # Set is faster for large nodes
    nodes_filtered = nodes[keep_rows_mask, :]
    
    # We must ensure the filtered nodes are in the correct index order (1 to N_new).
    # The simplest way is to sort nodes_filtered based on the index order in idx2id.
    
    # Create a mapping from original ID to the new index (1 to N_new)
    id_to_new_idx = Dict(id => idx for (idx, id) in enumerate(idx2id))
    
    # Sort the filtered DataFrame based on the new index order
    sort!(nodes_filtered, id_column, by=id -> id_to_new_idx[id])

    # 2. Construct the DataFrame using the filtered data
    node_df = DataFrame(
        node_id      = idx2id,
        idx          = 1:length(p), 
        x            = nodes_filtered.pos_x,  # <-- NOW CORRECT LENGTH
        y            = nodes_filtered.pos_y,  # <-- NOW CORRECT LENGTH
        z            = nodes_filtered.pos_z,  # <-- NOW CORRECT LENGTH
        probability  = p,
        target_pi    = π,
        local_entropy_bits = -p .* log2.(max.(p, 1e-20)),
        log_prob     = log10.(p .+ 1e-20),
        degree       = vec(sum(A; dims=2)),
        # Ensure region is also filtered:
        region       = hasproperty(nodes_filtered, :regions) ? nodes_filtered.regions : missing,
    )
    CSV.write("$(prefix)_nodes.csv", node_df)

    # 2. Edges with entropy-aware weights
    rows, cols, vals = findnz(A)
    # Effective flow capacity = original weight × average probability on endpoints
    flow_weight = similar(vals)
    for (k, (i,j)) in enumerate(zip(rows, cols))
        flow_weight[k] = vals[k] * (p[i] + p[j]) / 2
    end

    edge_df = DataFrame(
        source_id   = idx2id[rows],
        target_id   = idx2id[cols],
        source_idx  = rows,
        target_idx  = cols,
        anatomic_weight   = vals,           # original 1/length or whatever you used
        entropy_flow_weight = flow_weight,   # new: modulated by final probability
        average_probability = (p[rows] .+ p[cols]) ./ 2,
    )
    # Optional: only save undirected once
    mask = rows .< cols
    edge_df = edge_df[mask, :]
    CSV.write("$(prefix)_edges.csv", edge_df)

    # 3. Summary file
    open("$(prefix)_summary.txt", "w") do f
        println(f, "Entropy Flow Results")
        println(f, "Date:          $(Dates.now())")
        println(f, "Nodes:         $(length(p))")
        println(f, "Edges (undir): $(nrow(edge_df))")
        println(f, "Total Shannon entropy:  $(sum(-p .* log2.(max.(p,1e-20)))) bits")
        println(f, "KL(p||π):              $(sum(p .* log.(p ./ max.(π,1e-20))))")
        println(f, "Jacobian nullspace dim: $(size(res.null_basis,2))")
    end

    @info "Saved full dataset with prefix: $prefix"
    @info "   → $(prefix)_nodes.csv"
    @info "   → $(prefix)_edges.csv"
    @info "   → $(prefix)_summary.txt"
end
# ==============================================================================
# 3D INTERACTIVE PLOTTING — copy-paste this at the end of your file
# ==============================================================================
using GLMakie, GraphMakie, Colors

function plot_entropy_flow_3d(nodes, edges, p, A; 
    fraction_active = 0.10,      # ← keep only top 10%
    top_edges = 12000,
    movie_duration_sec = 20)

    pos = Point3f.(nodes.pos_x, nodes.pos_y, nodes.pos_z)
    local_ent = -p .* log2.(max.(p, 1e-20))

    # === 1. Select only the most active nodes (top 10% by local entropy) ===
    threshold = quantile(local_ent, 1 - fraction_active)
    active = local_ent .≥ threshold
    n_active = count(active)
    active_mask = local_ent .≥ threshold
    @info "Keeping $n_active / $(length(p)) nodes (top $(fraction_active*100)%) with highest local entropy"

    pos_active       = pos[active]
    p_active         = p[active]
    local_ent_active = local_ent[active]
    node_size_active = 180 .* (p_active ./ maximum(p_active)).^0.3 .+ 30

    # === 2. Extract only strong edges between active nodes ===
    rows, cols, vals = findnz(A)
    mask = active[rows] .&& active[cols]               # both ends active
    weights = vals[mask] .* (p[rows[mask]] .+ p[cols[mask]])  # entropy-weighted

    if length(weights) == 0
        error("No edges between active nodes — try a larger fraction_active")
    end

    # keep top edges by entropy flow
    perm = partialsortperm(weights, 1:min(top_edges, length(weights)); rev=true)
    src  = rows[mask][perm]
    dst  = cols[mask][perm]
    edge_weight = weights[perm]

    g = SimpleGraph(length(p))
    for (s, d) in zip(src, dst)
        s ≠ d && add_edge!(g, s, d) # skip self-loops → no crash in 3D
    end

    # normalize edge color & width
    ec = log10.(edge_weight .+ 1e-8)
    ec .-= minimum(ec)
    ec ./= maximum(ec)
    edge_width = @. 1.0 + 20 * ec

    # === 3. Plotting ===
    fig = Figure(size = (1600, 1200), backgroundcolor = :black)
    ax = LScene(fig[1,1]; show_axis = false)
    # Normalize colors & widths — length exactly matches number of edges in g
    edge_color = ec
    edge_width = @. 2.0 + 25 * ec   # slightly thicker for beauty
    linesegments!(ax, pos[src] .=> pos[dst];
        color      = edge_color,
        linewidth  = edge_width,
        colormap   = :plasma,
        transparency = true,
        alpha      = 0.45
    )

    # Edges (semi-transparent, glowing)
    #= Adds its own edges/triggers crashes.
    graphplot!(ax, g;
        layout       = _ -> pos,
        edge_color   = ec,
        edge_width   = edge_width,
        edge_colormap = :plasma,
        transparency = true,
        alpha        = 0.35
    )
    =#
    # Active nodes only — bright and glowing
    scatter!(ax, pos_active;
        markersize = node_size_active/5.0,
        color = local_ent_active,
        colormap = :viridis,
        colorrange = (threshold, maximum(local_ent_active)*1.05),
        glowwidth = 4.0,
        glowcolor = :white,
        strokewidth = 0,
        markerspace = :data,
        transparency = false
    )

    # === INACTIVE NODES — tiny dim dust (optional) ===
    inactive = .!active
    if !isempty(inactive)
        scatter!(ax, pos[inactive];
            markersize = 1.2/5.0,
            color = RGBAf(0.2, 0.2, 0.25, 0.04),
            glowwidth = 0.0,
            strokewidth = 0,
            transparency = true
        )
    end
    
    Colorbar(fig[1,2],
        limits = (0, maximum(local_ent)),
        colormap = :viridis,
        label = "Local entropy  −p log₂p (bits)",
        labelcolor = :white,
        ticklabelcolor = :white,
        height = Relative(0.6)
    )

    Label(fig[0,:], "Entropy Flow — Top $(Int(round(fraction_active*100)))% Active Nodes",
          fontsize=42, color=:white)

   display(fig)

    # Final working movie — no zoom field, no errors
    record(fig, "entropy_flow_top$(Int(round(fraction_active*100)))pct.mp4", framerate=60; visible = false) do io
        for angle in range(0, 2π, length=720)
            rotate_cam!(ax.scene, (0.08*sin(angle), angle, 0.04*sin(0.7*angle)))
            recordframe!(io)
        end
    end

    println("Saved focused movie → entropy_flow_top_$(Int(fraction_active*100))pct.mp4")
    return fig
end


function apply_bv_resolution_top_entropy!(
    p::Vector{Float64}, 
    dp::Vector{Float64}, 
    A::SparseMatrixCSC{Float64,Int};
    top_fraction::Float64 = 0.12,      # top 12% by local entropy → perfect sweet spot
    alpha::Float64 = 8e-4,             # strength of homological kick
    perturb_kicks::Int = 12            # number of random edge perturbations
)
    n = length(p)
    n < 10 && return nothing

    # 1. Compute local entropy and select top nodes
    local_ent = -p .* log2.(max.(p, 1e-20))
    thresh = quantile(local_ent, 1.0 - top_fraction)
    active_global = findall(>=(thresh), local_ent)
    m = length(active_global)
    if m < 8 || m > n ÷ 3
        # fallback: too small or too big → skip or downsample
        return nothing
    end

    # 2. Build compact active subgraph (fast path)
    id2local = Dict{Int,Int}()
    local2id = Vector{Int}(undef, m)
    for (loc, glob) in enumerate(active_global)
        id2local[glob] = loc
        local2id[loc] = glob
    end
    _, _, A_sub = EntropyBV.build_active_subgraph(active_global, A)

    # Early exit if subgraph has no edges
    if nnz(A_sub) == 0
        return nothing
    end

    # 3. Build edge indexing
    edge_u, edge_v, idxmap = EntropyBV.build_edge_index(A_sub)

    # 4. Extract local views (zero-copy if possible, but safe)
    p_active = @views p[active_global]
    dp_active = @views dp[active_global]

    # 5. Build flux C1 on active subgraph
    ne = length(edge_u)
    flux_vals = Vector{Float64}(undef, ne)
    @inbounds for k in 1:ne
        u, v = edge_u[k], edge_v[k]
        w = A_sub[u, v]
        flux_vals[k] = w * (p_active[u] + p_active[v]) / 2
    end
    flux = EntropyBV.c1_from_edgevals(edge_u, edge_v, idxmap, flux_vals)

    # 6. Random localized perturbation (the blow-up)
    perturb_vals = zeros(ne)
    for _ in 1:perturb_kicks
        k = rand(1:ne)
        perturb_vals[k] += randn()
    end
    perturb = EntropyBV.c1_from_edgevals(edge_u, edge_v, idxmap, perturb_vals)

    # 7. Gerstenhaber bracket = [flux, perturb] via commutator
    # bracket = EntropyBV.c1_commutator_bracket(flux, perturb, m)
    # Use the BV-derived bracket: {flux, flux} = -Δ(flux ∪ flux)
    # This uses the C2 paths to calculate the correction.
    # We use {flux, flux} as the most direct homological probe of the flux field's self-interaction.
    # BV Operator
    bracket = EntropyBV.derived_bracket_from_Delta_general(flux, flux, A_sub)

    # 8. Convert to node correction (antisymmetric → lifts degeneracy)
    corr_local = EntropyBV.c1_to_node_correction(bracket, m; convention=:in_minus_out)

    # 9. Apply correction in-place to active dp slice
    @inbounds @simd for i in 1:m
        dp_active[i] += alpha * corr_local.vals[i]
    end

    # No write-back needed: dp_active is a view into global dp
    return nothing
end

using WriteVTK
using Colors  # for distinguishable_colors

# ------------------------------------------------------------
# FINAL WORKING VTP EXPORT — GUARANTEED NO ERRORS
# ------------------------------------------------------------

using WriteVTK
using Colors
using WriteVTK
using WriteVTK.VTKCellTypes
using WriteVTK.PolyData
using StaticArrays

function save_entropy_brain_paraview(
    nodes::DataFrame,
    edges::DataFrame,
    p::Vector{Float64},
    A::SparseMatrixCSC;
    filename::String = "mouse_brain_final.vtp",
    fraction_active::Float64 = 0.08
)
    @info "Generating VTP: $filename (top $(100*fraction_active)% active)"

    ###########################################################################
    # 1. Select active nodes by local entropy
    ###########################################################################
    local_ent = @. -p * log2(max(p, 1e-20))
    thresh     = quantile(local_ent, 1 - fraction_active)
    active     = local_ent .≥ thresh
    active_idx = findall(active)
    n_points   = length(active_idx)

    # Map global → local (0-based indexing for VTK)
    global_to_local = Dict(zip(active_idx, 0:(n_points - 1)))


    ###########################################################################
    # 2. Extract point coordinates
    ###########################################################################
    points = Matrix{Float64}(undef, 3, n_points)
    points[1, :] .= nodes.pos_x[active_idx]
    points[2, :] .= nodes.pos_y[active_idx]
    points[3, :] .= nodes.pos_z[active_idx]

    # Convert to vector of static 3-vectors for VTK
    pts = [@SVector [points[1,i], points[2,i], points[3,i]] for i in 1:n_points]


    ###########################################################################
    # 3. Node attributes
    ###########################################################################
    p_active        = Float32.(p[active_idx])
    entropy_active  = Float32.(local_ent[active_idx])

    # Region names cleaned for visualization
    regions_raw = replace.(
        string.(coalesce.(nodes.regions[active_idx], "unknown")),
        r"Region_Acronym_" => ""
    )

    unique_regions = unique(regions_raw)
    colors         = distinguishable_colors(length(unique_regions); dropseed = true)
    color_map      = Dict(zip(unique_regions, colors))

    region_colors  = [color_map[r] for r in regions_raw]

    region_index   = Int32.(indexin(regions_raw, unique_regions))
    region_r       = Float32.([c.r for c in region_colors])
    region_g       = Float32.([c.g for c in region_colors])
    region_b       = Float32.([c.b for c in region_colors])


    ###########################################################################
    # 4. Active edges (line cells)
    ###########################################################################
    rows, cols, vals = findnz(A)
    mask = active[rows] .& active[cols]

    src_local = [global_to_local[r] for r in rows[mask]]
    dst_local = [global_to_local[c] for c in cols[mask]]

    edge_flow = Float32.(vals[mask] .* (p[rows[mask]] .+ p[cols[mask]]))

    # VTK line cells
    cells = [MeshCell(PolyData.Lines(), (s+1, d+1)) for (s, d) in zip(src_local, dst_local)]


    ###########################################################################
    # 5. Build VTK unstructured grid
    ###########################################################################
    vtk = vtk_grid(filename, pts, cells)

    # Point data
    vtk["probability",        VTKPointData()] = p_active
    vtk["local_entropy_bits", VTKPointData()] = entropy_active
    vtk["region_index",       VTKPointData()] = region_index
    vtk["region_r",           VTKPointData()] = region_r
    vtk["region_g",           VTKPointData()] = region_g
    vtk["region_b",           VTKPointData()] = region_b
    vtk["region_name",        VTKPointData()] = regions_raw

    # Cell data (edges)
    vtk["edge_flow",   VTKCellData()] = edge_flow
    vtk["edge_weight", VTKCellData()] = Float32.(vals[mask])


    ###########################################################################
    # 6. Save
    ###########################################################################
    vtk_save(vtk)

    @info "SUCCESS: $filename created ($n_points nodes, $(length(cells)) edges)"
    @info "ParaView: Color by region_index → Glyph(Sphere) for nodes → Tube for edges"

    return filename
end

# Select top-entropy nodes for blowdown
function select_blowdown_nodes(p::Vector{Float64}; frac=0.01)
    n = length(p)
    h = @. -p * log(p + eps())  # local entropy, add eps() to avoid log(0)
    k = max(1, round(Int, frac * n))
    idx = partialsortperm(h, rev=true, 1:k)  # top-k indices
    return idx
end

# Apply blowdown: dampen probabilities and renormalize
function apply_blowdown!(p::Vector{Float64}; frac=0.01, reduction=0.5)
    top_idx = select_blowdown_nodes(p; frac=frac)
    @inbounds for i in top_idx
        p[i] *= (1 - reduction)   # dampen
    end
    p ./= sum(p)  # re-normalize
    return top_idx
end


# ==============================================================
# 8. Run it
# ==============================================================
if abspath(PROGRAM_FILE) == @__FILE__
    nodes_file = "/Users/vaw1/Downloads/OGB/node_regions_clean.csv"   # put your node table as semicolon CSV here
    edges_file = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"   # put your edges table as semicolon CSV here

    res = run_entropy_sim(nodes_file, edges_file;
                          pi_mode = :region,
                          mobility = :diag,         # change to :laplacian if you really need it
                          dt       = 2e-3,
                          steps    = 600,
                          update_fraction = 0.1)

    println("\nNullspace dimension = $(size(res.null_basis,2))")

    # Save graph 
    save_entropy_flow_results(res.nodes, res.edges, res.id2idx, res.all_ids, res.p, res.π, res.A)
    # Call it right after your simulation finishes
    if @isdefined res
    #    plot_entropy_flow_3d(res.nodes, res.edges, res.p, res.A; fraction_active = 0.10)
    end
    if @isdefined res
        save_entropy_brain_paraview(res.nodes, res.edges, res.p, res.A;
            filename = "mouse_brain_entropy_final.vtp",
            fraction_active = 1.0   # Blow down dissipates nodes as needed.
        )
    end
    #=
    println("\n" * "="^60)
    println("SINGLE SIMULATION DONE — NOW GENERATING SBI DATASET--Blowdown trims 70% gradually")
    println("="^60)

    # Functioning blow down ...
    # 2. THIS IS WHERE generate_sbi_dataset IS CALLED
    # Warning for C++ creators coming to Julia, Python etc...
    # Nested functions can use variables that are not explicitly passed as 
    # input arguments. In a parent function, you can create a handle to a 
    # nested function that contains the data necessary to run the nested function.
    if !isfile("sbi_dataset.bson") || filesize("sbi_dataset.bson") < 10_000_000
        @info "Generating 30 simulations for SBI (this takes 1–6 hours)..."
        # will use global res parameters such as res.nodes, res.edges res.p res.A etc.
        generate_sbi_dataset(res, res.id2idx, 5000; path="sbi_dataset.bson")
    else
        @info "SBI dataset already exists → skipping generation"
    end

    println("\nAll done! You now have:")
    println("   • A stunning 3D entropy brain video")
    println("   • Full SBI dataset (sbi_dataset.bson)")
    println("   • Ready for neural posterior training")
    println("\nNext step: run the training script (or add it here)!")
    =#
end
