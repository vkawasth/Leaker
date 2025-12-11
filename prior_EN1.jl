using BSON
using CSV, DataFrames
using Flux
using Random
using Statistics
using UMAP
using Plots
using LinearAlgebra

OUTCOMES = [
    :epilepsy, :confusion, :blurred_vision, :sweating,
    :coma, :energized_alert, :hyperactivity, :anxiety
]

# --- CONFIGURATION FROM USER INPUT ---
NODES_FILE = "/Users/vaw1/Downloads/OGB/node_regions_clean.csv"
DATA_FILE = "sbi_dataset_top1percentProb_BV.bson" # Contains 'sims' and 'θs'
TARGET_OUTCOME = :anxiety              # The outcome we want to blame nodes for
const METADATA_CSV_PATH = "entropy_flow_final_nodes_blowdown.csv" 

struct EntropyPriorParams
    cortex_scale::Float64      # 5–25
    hippo_scale::Float64       # 8–30
    sensory_scale::Float64     # 3–15
    cerebellum_scale::Float64  # 0.1–3
    noise::Float64             # 0.01–0.5
end

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

# -----------------------------------------
# 1. Load blown-down probability dataset
# -----------------------------------------
data = BSON.load(DATA_FILE)
# data[:sims] should contain:
#   p, top_entropy.idx, top_entropy.p_top, top_entropy.h_top, outcomes
samples = data[:sims]
priors = data[:θs]
# -----------------------------------------
# 2. Load node→region map
# -----------------------------------------
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
region_map = load_node_region_map(NODES_FILE)



nsamples = length(samples)
initial_bias = -1.0

# -----------------------------------------
# 3. Build training table for a given outcome
# -----------------------------------------
function build_training_table(outcome_symbol)
    outcome_index = findfirst(==(outcome_symbol), OUTCOMES)

    rows = Vector{NamedTuple}()

    for s = 1:nsamples
        ptop  = samples[s].top_entropy.p_top
        htop  = samples[s].top_entropy.h_top
        idxs  = samples[s].top_entropy.idx
        y     = samples[s].outcomes[outcome_index]  # 0 or 1 for this sample

        for j in eachindex(idxs)
            node = idxs[j]
            push!(rows, (
                node=node,
                p=ptop[j],
                h=htop[j],
                label=y
            ))
        end
    end

    return DataFrame(rows)
end

# build training data for epilepsy
df_epi = build_training_table(:epilepsy)

# -----------------------------------------
# 4. Convert to tensors, include region priors
# -----------------------------------------
function to_tensor_with_prior(df::DataFrame, region_map::Dict{Int, Vector{String}}, prior_map::Dict{String, Float64})
    N = nrow(df)
    X = zeros(Float32, 3, N)   # 2 features + 1 prior
    y = Float32.(df.label)     # N-element vector

    # --- Feature Filling (Unchanged) ---
    for (i, row) in enumerate(eachrow(df))
        node = row.node
        X[1, i] = Float32(row.p)
        X[2, i] = Float32(row.h)

        # Use the first region in region_map[node] to get prior
        regs = get(region_map, node, [""])  # fallback to [""] if node not found
        prior_val = isempty(regs) ? 1.0 : get(prior_map, regs[1], 1.0)
        X[3, i] = Float32(prior_val)
    end

    # --- Normalization and STATS CAPTURE (New) ---
    # Initialize vectors to store the mean and standard deviation for each feature (row)
    means = zeros(Float32, 3)
    stds = zeros(Float32, 3)

    # Normalize each row (feature) and save the calculated statistics
    for j in 1:size(X,1)
        # 1. Capture the statistics from the training data
        means[j] = mean(X[j, :])
        stds[j] = std(X[j, :])
        
        # 2. Apply normalization (using the captured stats)
        X[j, :] .= (X[j, :] .- means[j]) ./ (stds[j] + 1e-6)
    end

    # --- Return Normalized Data AND Stats (Fixed Output) ---
    # The statistics (means, stds) are now returned alongside the features and labels
    return X, reshape(y, 1, :), means, stds 
end

X_epi, y_epi = to_tensor_with_prior(df_epi, region_map, MOUSE_REGION_PRIOR)
#Plot it
# --- Prepare data ---
# X_epi is 3 x N (features x samples)

# ----------------------
# Take a small subset
# ----------------------
N_total = size(X_epi, 2)
N_subset = min(1000, N_total)         
idxs = randperm(N_total)[1:N_subset]

# X_umap_input is (Features x Samples) - required for UMAP.jl
X_umap_input = Float32.(X_epi[:, idxs]) # X_epi is (F x N), so no transpose needed on X_epi[:, idxs]
y_umap = vec(y_epi[1, idxs])        # 1D labels vector

# ----------------------
# Fit UMAP (FIXED SYNTAX and INIT KEYWORD)
# ----------------------
umap_model = UMAP.UMAP_(
    X_umap_input, # Positional Arg 1: Data Matrix (3 x N_subset)
    2;            # Positional Arg 2: Output Dimension
    n_neighbors = 5,
    min_dist = 0.1,
    init = :random # <--- FIXED: Use :random for faster initialization
)

# Learn embedding (fit! is implied by the constructor when passing X)
# Get embedding (Embedding is in (Features x Samples) format: 2 x N_subset)
embedding_fxn = UMAP.transform(umap_model, X_umap_input)

# ----------------------
# Plot
# ----------------------
p = scatter(
        embedding_fxn[1, :],   # UMAP1 coordinates (Row 1)
        embedding_fxn[2, :],   # UMAP2 coordinates (Row 2)
        group=y_umap,          # Use the 1D labels vector
        title="UMAP of Node Features Colored by Outcome",
        xlabel="UMAP1", ylabel="UMAP2",
        markersize=6,
        legend=:topright,
        palette=[:blue, :red]
    )
display(p)
savefig("umap_plot.png")



X_umap = transpose(X_epi)  # N x 3, rows = samples


# Standardize features
for j in 1:size(X_umap, 2)
    X_umap[:, j] .= (X_umap[:, j] .- mean(X_umap[:, j])) ./ (std(X_umap[:, j]) + 1e-6)
end

# Labels (0 or 1)
labels = vec(y_epi)  # make 1D vector


X_epi, y_epi = to_tensor_with_prior(df_epi, region_map, MOUSE_REGION_PRIOR)        
# 1. Calculate N0 and N1 (assuming y_epi is 1xN matrix of 0s and 1s)
N1 = sum(y_epi)
N0 = length(y_epi) - N1
N_total = length(y_epi)

# W1 and W0 are the weights for the positive (1) and negative (0) classes
const W1 = N_total / (2 * N1)
const W0 = N_total / (2 * N0)

println("Class weights: W0 (Negative) = $(round(W0, digits=2)), W1 (Positive) = $(round(W1, digits=2))")

# 2. Define the weighted loss function
function weighted_binarycrossentropy(y_pred::AbstractArray{T}, y_true::AbstractArray{T}, w1::Real, w0::Real) where T<:AbstractFloat    # Weight map: W1 for positive (1), W0 for negative (0)
    # Since y_true is a 1xN matrix of 0s and 1s:
    # y_true .* w1 sets positive weights to w1, negative to 0.
    # (1 .- y_true) .* w0 sets negative weights to w0, positive to 0.
    w = y_true .* w1 .+ (1.0f0 .- y_true) .* w0

    # Element-wise weighted binary crossentropy
    return -mean(w .* (y_true .* log.(y_pred .+ 1e-8) .+ (1.0f0 .- y_true) .* log.(1.0f0 .- y_pred .+ 1e-8)))
end


# -----------------------------------------
# 5. Define model
# -----------------------------------------
# Calculate the log-odds (logit) for the desired starting probability (P)
# Let's say P = 0.28 (estimated positive fraction)
# Logit = log(P / (1 - P)) = log(0.28 / 0.72) ≈ -0.94
initial_bias = -1.0 # Use -1.0 as a safe bias towards the negative class
# -----------------------------------------
# 5. Update model for 3 input features
# -----------------------------------------
model = Chain(
    Dense(3, 64, leakyrelu),   # now 3 inputs
    Dense(64, 32, leakyrelu),
    Dense(32, 16, relu),   # Third layer
    # Add an explicit initial bias (b) to push the initial prediction down
    Dense(16, 1, bias=fill(initial_bias, 1)), # initial bias is now -1.0
    σ
)

loss_unweighted(m) = Flux.binarycrossentropy(model(x), y)
opt = ADAM(1e-3)
#reshape y
y_epi = reshape(y_epi, 1, :)
# -----------------------------------------
# 6. Train with class weighting
# -----------------------------------------
# -------------------------------
# 1. Training loop (Flux 0.13+)
# -------------------------------
function train_model!(model::Chain, X::Matrix{Float32}, y::Matrix{Float32};
    epochs::Int=500, lr::Float64=1e-3, verbose::Bool=true)

    # --- Define loss function that takes a model `m` ---
    loss(m) = Flux.binarycrossentropy(m(X), y) 
    # 3. Use this new loss in your training loop:
    loss_weighted(m) = weighted_binarycrossentropy(m(X), y, Float32(W1), Float32(W0))
    # --- Set up optimizer with state ---
    opt = Flux.Adam(lr)
    state = Flux.setup(opt, model)   # create optimizer state

    for epoch in 1:epochs
        # Corrected: Take the gradient of `loss(m)` with respect to `m`.
        # Zygote will automatically only trace the trainable parts.
        grads = Flux.gradient(loss_weighted, model) # <--- **THIS IS THE CRITICAL CHANGE**

        # Update model parameters using optimizer state
        Flux.update!(state, model, grads[1]) # grads is now a tuple: (gradient_of_model, )

       
        if  verbose &&  epoch % 20 == 0
            # Get the gradient of the first layer's weights
            grad_w = grads[1].layers[1].weight
            println("Epoch $epoch → Max Abs Grad (Layer 1): ", maximum(abs, grad_w))
            println("Epoch $epoch → Loss = ", loss(model))
        end
        
    end

    return model
end

model = train_model!(model, X_epi, y_epi, epochs=500, lr=1e-2)  # fewer epochs for test

# -----------------------------------------
# 7. Inference: Score all nodes for epilepsy
# -----------------------------------------
function score_nodes(model, df, region_map, prior_map, X_means::Vector{Float32}, X_stds::Vector{Float32})
    N = nrow(df)
    X = zeros(Float32, 3, N)

    for (i, row) in enumerate(eachrow(df))
        node = row.node
        X[1, i] = Float32(row.p)
        X[2, i] = Float32(row.h)

        regs = get(region_map, node, [""])
        prior_val = isempty(regs) ? 1.0 : get(prior_map, regs[1], 1.0)
        X[3, i] = Float32(prior_val)
    end

    # Normalize each row using the TRAINING STATISTICS
    for j in 1:size(X,1)
        X[j, :] .= (X[j, :] .- X_means[j]) ./ (X_stds[j] + 1e-6) # <--- FIXED
    end

    scores = model(X)
    df.score = vec(scores)
    return df
end


df_scored = score_nodes(model, df_epi, region_map, MOUSE_REGION_PRIOR, X_means, X_stds)
println("Positive fraction: ", mean(df_epi.label))
# -----------------------------------------
# 8. Pick top-K nodes
# -----------------------------------------
function topK(df, K)
    sort(df, :score, rev=true)[1:K, :]
end

top200 = topK(df_scored, 200)

# -----------------------------------------
# 9. Map nodes → regions
# -----------------------------------------
top200[:, :regions] = [region_map[n] for n in top200.node]
for i in 1:nrow(top200)
    println("Node ", top200.node[i], " → Region(s) ", top200.regions[i],
            " → Score ", round(top200.score[i], digits=6))
end