using BSON
using StatsBase
using LinearAlgebra
using CSV
using DataFrames

NODES = "/Users/vaw1/Downloads/OGB/node_regions_clean.csv"   # put your node table as semicolon CSV here
#DATA = "sbi_1storder_dataset.bson"
DATA = "sbi_2ndorder_dataset.bson"

struct EntropyPriorParams
    cortex_scale::Float64      # 5–25
    hippo_scale::Float64       # 8–30
    sensory_scale::Float64     # 3–15
    cerebellum_scale::Float64  # 0.1–3
    noise::Float64             # 0.01–0.5
end
#=
data = BSON.load("sbi_dataset.bson")

println("TYPE: ", typeof(data))
println("FIELDS / KEYS:")

println(typeof(data[:θs]))
println(typeof(data[:sims]))

println("θs length: ", length(data[:θs]))
println("sims length: ", length(data[:sims]))

println("θs sample:")
sims = data[:sims]
param = data[:θs]
println(param)

function preview_sims(sims; n=3)
    for i in 1:min(n, length(sims))
        sim = sims[i]
        println("\n=== Simulation $i ===")
        println("\nLength p: ", length(sim.p))
        println("\nLength Top Ent : ", length(sim.top_entropy))
        println("\nLength Top Ent Idx : ", length(sim.top_entropy.idx))
        println("\nLength Top Ent p_top : ", length(sim.top_entropy.p_top))
        println("\nLength Top Ent h_top : ", length(sim.top_entropy.h_top))
        println("p[1:5]               = ", sim.p[1:5])
        println("top_entropy.idx[1:5] = ", sim.top_entropy.idx[1:5])
        println("top_entropy.p_top[1:5] = ", sim.top_entropy.p_top[1:5])
        println("top_entropy.h_top[1:5] = ", sim.top_entropy.h_top[1:5])
        println("outcomes             = ", sim.outcomes)
    end
end
preview_sims(sims, n=3)
=#
# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

const OUTCOMES = [
    :epilepsy, :confusion, :blurred_vision, :sweating,
    :coma, :energized_alert, :hyperactivity, :anxiety
]

# Provided by you
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

const OUTCOMES = [:epilepsy, :confusion, :blurred_vision, :sweating, :coma, :energized_alert, :hyperactivity, :anxiety]

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
const REGION_TO_MACRO = Dict{String,Symbol}()

for r in [
    "ACA","AI","AUD","ECT","FRP","GU","ILA","MO","ORB","PERI",
    "PL","POST","PRE","RSP","SS","TEa","VIS","VISC"
]
    REGION_TO_MACRO[r] = :cortex
end

for r in ["CA1sp","SUB","HPF"]
    REGION_TO_MACRO[r] = :hippocampus
end

for r in ["AOB","AOBgr","AON","PIR","TT","COA","PAA"]
    REGION_TO_MACRO[r] = :sensory
end

for r in ["CB","CBXmo"]
    REGION_TO_MACRO[r] = :cerebellum
end

# default
REGION_TO_MACRO[""] = :cortex

"""
node_region_map :: Dict{Int, Vector{String}}
Maps node_id → list_of_regions
"""
function load_node_region_map(path::String)
    df = CSV.read(path, DataFrame; delim=';', ignorerepeated=true)

    d = Dict{Int, Vector{String}}()

    for row in eachrow(df)
        id = Int(row[1])
        reg_raw = String(row[end])   # column with ['REGION'] text

        # Clean: remove brackets, quotes, spaces
        clean = replace(reg_raw, ['[',']','\'','"'] => "")
        regs = strip.(split(clean, ','))

        d[id] = regs
    end

    return d
end

# ---------------------------------------------------------
# SIMULATION-BASED INFERENCE
# ---------------------------------------------------------

"""
Compute region activation score based on entropy weights.
h_top is already high-entropy (top 1%), so treat as importance weights.

Returns Dict{String, Float64} mapping region → total_weight.
"""
function compute_region_activation(
    idx::AbstractVector{<:Integer},
    h_top::AbstractVector{<:Real},
    node_region_map::Dict{Int, Vector{String}}
)

    macro_scores = Dict(
        :cortex => 0.0,
        :hippocampus => 0.0,
        :sensory => 0.0,
        :cerebellum => 0.0
    )

    for (nid, h) in zip(idx, h_top)
        regions = get(node_region_map, nid, String[])
        for r in regions
            macro_group = get(REGION_TO_MACRO, r, :cortex)  # default cortex or choose :unknown
            macro_scores[macro_group] += h
        end
    end

    return macro_scores
end


"""
Given region_activation_scores, compute how compatible the sample is
with each OUTCOME (epilepsy, confusion, …)

Uses simple cosine similarity between:
  vector_of_region_scores   vs.   indicator_vector_of_expected_regions
"""
f# Main scoring function: return per-outcome scores
function score_outcomes(
    region_scores::Dict{String,Float64},
    outcome_mask::Vector{Int};
    priors::Vector{EntropyPriorParams}
)

    scores = Dict{Symbol,Float64}()

    @assert length(outcome_mask) == length(OUTCOMES) "Outcome mask must match OUTCOMES length"

    # Compute weighted region activation once
    weighted = 0.0
    for p in priors
        weighted +=
            (region_scores["cortex"]      * p.cortex_scale)    +
            (region_scores["hippocampus"] * p.hippo_scale)     +
            (region_scores["sensory"]     * p.sensory_scale)   +
            (region_scores["cerebellum"]  * p.cerebellum_scale)
    end
    weighted /= length(priors)  # mean activation

    # Now score EACH outcome independently
    for (i, outcome_bit) in enumerate(outcome_mask)
        p = priors[i]   # outcome-specific parameters

        # logistic probability
        prob = 1.0 / (1.0 + exp(-(weighted + p.noise)))

        # assign score
        score_i = outcome_bit == 1 ? log(prob) : log(1 - prob)

        scores[OUTCOMES[i]] = Float64(score_i)  # ensure Float64
    end

    return scores
end


# String-key fallback
function score_outcomes(
    region_scores::Dict{Symbol, Float64},
    outcomes::Vector{Int};
    priors
)
    region_scores_str = Dict(String(k) => v for (k,v) in region_scores)
    return score_outcomes(region_scores_str, outcomes; priors=priors)
end

# small, safe softmax implementation
function softmax_vec(v::AbstractVector{<:Real})
    v = Float64.(v)
    vmax = maximum(v)
    exps = exp.(v .- vmax)
    exps ./ sum(exps)
end

function cosine_similarity(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    da = sum(a .^ 2)
    db = sum(b .^ 2)
    if da == 0 || db == 0
        return 0.0
    end
    return (dot(a, b)) / sqrt(da * db)
end

# ---------------------------------------------------------
# PIPELINE FOR ONE SAMPLE
# ---------------------------------------------------------

function infer_sample(sample, node_region_map; priors=nothing)

    # unpack
    p    = sample.p
    top  = sample.top_entropy  # has idx, h_top, p_top
    

    # 1. Compute region activation from top-entropy nodes
    region_scores = compute_region_activation(
        top.idx, top.h_top, node_region_map
    )

    # 2. Score outcomes (with priors)
    outcome_scores = score_outcomes(region_scores, sample.outcomes; priors)

    # 3. Turn scores into normalized posterior
    vals = collect(values(outcome_scores))
    post = softmax_vec(vals)
    posterior = Dict(OUTCOMES[i] => post[i] for i in 1:length(OUTCOMES))
    # convert posterior dict → vector in OUTCOMES order
    posterior_vec = [posterior[o] for o in OUTCOMES]
    # how are we matching
    cosine_match = cosine_similarity(posterior_vec, sample.outcomes)
    return (
        posterior       = posterior,       # predicted outcome probabilities
        outcome_scores  = outcome_scores,  # raw scores before softmax
        region_scores   = region_scores,   # intermediate values
        cosine_match    = cosine_match,
        p               = p,               # full 3.5M node probability vector
        top_entropy     = top,              # (idx, h_top, p_top)
        outcomes         = sample.outcomes
    )
end

# ---------------------------------------------------------
# TOP-LEVEL: PROCESS A WHOLE BSON FILE
# ---------------------------------------------------------

"""
Run SBI-style inference over all 30 samples inside the BSON.
"""
function simulate_bson(path_bson::String, path_nodes::String)
    data = BSON.load(path_bson)

    sims   = data[:sims]      # Vector of NamedTuples
    priors = data[:θs]        # Vector{EntropyPriorParams}

    node_region_map = load_node_region_map(path_nodes)

    # Pass priors vector into each inference call
    results = [
        infer_sample(sim, node_region_map;
                     priors = priors)   # or pass nodes_df if needed
        for sim in sims
    ]

    return results
end

result = simulate_bson(DATA, NODES)
for (i, r) in enumerate(result)
    println("Sample $i posterior:")
    println(r.posterior, ", ", r.cosine_match, ", ",r.outcomes)
end
