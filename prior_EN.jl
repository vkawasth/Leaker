using BSON
using DataFrames
using LinearAlgebra
using StatsBase
using CSV
using Random
using GLM
using BSON
using DataFrames
using LinearAlgebra
using StatsBase
using Random
using SparseArrays
using DataStructures
using StanBase
using GLMNet # Required for efficient Sparse Lasso/Elastic Net fitting
using Printf # For formatted output
#using RegularizedLeastSquares
using Statistics # Required for mean and std
using SparseArrays # For SparseMatrixCSC type
using MLJ
using MLJModels
using MLJLinearModels # Provides the ElasticNet model
using StatsBase
using ProximalAlgorithms

# NOTE: To run this code, you must install and use a package that supports
# Elastic Net/Lasso regression, such as 'GLM.jl' or a dedicated package
# like 'Lasso.jl' (or ScikitLearn.jl if using that ecosystem).
# For demonstration, we will assume a conceptual 'ElasticNet' function exists.
# Pkg.add(["CSV", "DataFrames", "BSON", "LinearAlgebra", "StatsBase"])

struct EntropyPriorParams
    cortex_scale::Float64      # 5–25
    hippo_scale::Float64       # 8–30
    sensory_scale::Float64     # 3–15
    cerebellum_scale::Float64  # 0.1–3
    noise::Float64             # 0.01–0.5
end

# --- CONFIGURATION FROM USER INPUT ---
NODES_FILE = "/Users/vaw1/Downloads/OGB/node_regions_clean.csv"
DATA_FILE = "sbi_2ndorder_dataset_blowdown.bson" # Contains 'sims' and 'θs'
TARGET_OUTCOME = :anxiety              # The outcome we want to blame nodes for
TOP_N_NODES = 10                       # The size of the "exceptional locus"

# Place this constant at the top of your script
const METADATA_CSV_PATH = "entropy_flow_final_nodes_2ndorder.csv" 
# Assuming DATA_FILE and other constants are already defined

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
# HELPER FUNCTIONS (Assuming data structures from user's input)
# ---------------------------------------------------------

"""
1. Extracts the full node activity matrix (X), the target outcome vector (Y),
   and the unique set of nodes flagged as high-entropy during simulation (sim_exceptional_nodes).
   X: [N_samples x N_nodes] matrix (from sim.p)
   Y: [N_samples x 1] vector (from sim.outcomes[anxiety_index])
   sim_exceptional_nodes: Set of 1-indexed node IDs that appeared in sim.top_entropy.idx
"""
function prepare_regression_data(data_file::String, target_outcome::Symbol, outcomes::Vector{Symbol})
    data = BSON.load(data_file)
    sims = data[:sims]

    # Find the index of the target outcome (anxiety)
    target_idx = findfirst(==(target_outcome), outcomes)
    if isnothing(target_idx)
        error("Target outcome $target_outcome not found in OUTCOMES list.")
    end

    # 1. Build the Design Matrix X (Node Activities)
    X_list = [sim.p for sim in sims]
    X = reduce(hcat, X_list)'
    N_samples, N_nodes = size(X)
    
    # 2. Build the Target Vector Y (Anxiety Outcome)
    Y = Float64[sim.outcomes[target_idx] for sim in sims]
    
    # 3. Identify the Unique Nodes Flagged as Exceptional During Simulation
    # Concatenate all 'top_entropy.idx' arrays from all simulations and get the unique set.
    # We check for hasproperty(:top_entropy) for robustness.

    # top entropy is really lowest entropy data (name was tagged wrong) -- Most stabledata See line 953 
    # top_core_data (max probability stable core is sent as top_entropy --)
    all_sim_exceptional_indices = vcat([s.top_entropy.idx for s in sims if hasproperty(s, :top_entropy)]...)
    sim_exceptional_nodes = Set(all_sim_exceptional_indices)
    
    println("Data Prepared:")
    println("  N_samples: $N_samples")
    println("  N_nodes: $N_nodes (The number of features to regress)")
    println("  Unique nodes flagged as high-entropy during sim (Simulated Locus Size): $(length(sim_exceptional_nodes))")
    
    return X, Y, N_nodes, sim_exceptional_nodes
end

# --- CONFIGURATION (from user input) ---
#const DATA_FILE = "sbi_2ndorder_dataset_blowdown.bson" # Contains 'sims' and 'θs'
#const TARGET_OUTCOME = :anxiety              # The outcome we want to blame nodes for
#const TOP_N_NODES = 10                       # The size of the final "exceptional locus" to report

# Parameters for the scalable analysis pipeline
const LASSO_ALPHA = 1.0                      # 1.0 for pure Lasso, 0.5 for Elastic Net
const TARGET_K_FOR_DETAIL = 1000             # The K value for which we perform the detailed Lasso analysis
const SAMPLE_SIZE = 10                       # Number of Node IDs to print as a sample
const FEATURE_COUNT_CANDIDATES = [200, 500, 1000, 2000, 5000] 

const OUTCOMES = [
    :epilepsy, :confusion, :blurred_vision, :sweating,
    :coma, :energized_alert, :hyperactivity, :anxiety
]

# --- DATA STRUCTURES ---

# Mutable struct for convenience in mapping coefficients
mutable struct NodeCoeffs
    H_coeff::Float64
    # We will use the main activity P as the single feature, so P_coeff is the only one needed
    P_coeff::Float64 
    NodeCoeffs() = new(0.0, 0.0)
end

# --- CORE SCALABILITY FUNCTIONS (Feature Filtering & Sparse Matrix) ---

"""
#Pass 1: Identifies the Top K nodes by total activity magnitude across all simulations.
#This is the critical dimensionality reduction step (N_nodes -> K).
"""
function identify_top_K_nodes(all_sims::Vector, k_final::Int)
    node_activity_sum = DefaultDict{Int, Float64}(0.0)
    
    # Aggregate activity for all nodes in one pass
    for sim in all_sims
        for (node_id, p_val) in enumerate(sim.p)
            # Use absolute activity as proxy for importance
            node_activity_sum[node_id] += abs(p_val) 
        end
    end
    
    # Sort and Select Top K
    sorted_nodes = sort([p for p in node_activity_sum], by=x -> x[2], rev=true)
    
    K_actual = min(k_final, length(sorted_nodes))
    top_k_indices = [item[1] for item in sorted_nodes[1:K_actual]]
    
    # Create Feature Mapping (Node Index -> Feature Column 1 to K_actual)
    idx_to_col = Dict{Int, Int}(idx => i for (i, idx) in enumerate(top_k_indices))
    
    return top_k_indices, K_actual, idx_to_col
end

function identify_top_K_nodes_by_variance(all_sims::Vector, k_final::Int)
    
    # 1. Prepare the full raw activity matrix X
    X_list = [sim.p for sim in all_sims]
    X = reduce(hcat, X_list)' # X: [N_samples x N_nodes]
    N_samples, N_nodes = size(X)

    # 2. Calculate the Standard Deviation for each node (feature)
    # We use std() across dimension 1 (samples) to get the variation per node.
    # The result is a 1xN_nodes array.
    node_std_devs = std(X, dims=1) 
    
    # 3. Create a collection of (node_id, std_dev) pairs
    node_metrics = Dict{Int, Float64}()
    for j in 1:N_nodes
        # Use node_id (1-indexed) and its standard deviation
        node_metrics[j] = node_std_devs[j]
    end
    
    # 4. Sort and Select Top K
    sorted_nodes = sort([p for p in node_metrics], by=x -> x[2], rev=true)
    
    K_actual = min(k_final, length(sorted_nodes))
    top_k_indices = [item[1] for item in sorted_nodes[1:K_actual]]
    
    # Create Feature Mapping (Node Index -> Feature Column 1 to K_actual)
    idx_to_col = Dict{Int, Int}(idx => i for (i, idx) in enumerate(top_k_indices))
    
    return top_k_indices, K_actual, idx_to_col
end

"""
#Pass 2: Constructs the Sparse Feature Matrix (X) using only the Top K features.
"""
# Three sparsification modes: :per_sample_topk, :per_feature_percentile, :per_feature_std
function prepare_sparse_regression_data(
    all_sims::Vector,
    target_outcome::Symbol,
    outcomes::Vector{Symbol},
    k_final::Int;
    mode::Symbol = :per_sample_topk,          # :per_sample_topk | :per_feature_percentile | :per_feature_std
    topk_per_sample::Int = 10,                # used if mode == :per_sample_topk
    percentile_keep::Float64 = 0.90,          # used if mode == :per_feature_percentile (0.0..1.0)
    std_multiplier::Float64 = 1.0,            # used if mode == :per_feature_std
    absolute_min::Float64 = 1e-9              # absolute floor to consider nonzero
)
    target_idx = findfirst(==(target_outcome), outcomes)
    if isnothing(target_idx)
        error("Target outcome $target_outcome not found in OUTCOMES")
    end

    N_samples = length(all_sims)
    N_nodes = length(all_sims[1].p)

    # 1) identify top K by variance (same as before)
    top_k_indices, K_actual, idx_to_col = identify_top_K_nodes_by_variance(all_sims, k_final)

    # 2) Build a raw dense-ish representation only for the selected K columns
    #    but we will not materialize a full dense N x K if we can avoid it.
    #    Instead, we compute candidate nonzeros according to chosen sparsification mode.

    # Prepare array-of-arrays for candidate (i, j, v) before sparse()
    I = Int[]
    J = Int[]
    V = Float64[]
    Y = zeros(Int, N_samples)

    # Build helper: column values (only for selected columns)
    # We'll create a vector of N_samples Float64 for each selected node (if needed by mode)
    if mode == :per_feature_percentile || mode == :per_feature_std
        # Precompute per-column vectors for selected nodes
        col_values = Dict{Int, Vector{Float64}}()
        for (col_idx, node_id) in enumerate(top_k_indices)
            v = Float64[sim.p[node_id] for sim in all_sims]
            col_values[node_id] = v
        end

        # If percentiles: compute cutoff per column
        col_cutoff = Dict{Int, Float64}()
        if mode == :per_feature_percentile
            
            for node_id in top_k_indices
                vals = abs.(col_values[node_id])
                cutoff = quantile(vals, percentile_keep) # keeps top (1 - percentile_keep)?? careful
                # Note: percentile_keep is the quantile value (e.g., 0.90 -> 90th percentile)
                # We'll keep abs(value) >= cutoff and > absolute_min
                col_cutoff[node_id] = max(cutoff, absolute_min)
            end
        else
            # std-based cutoff
            for node_id in top_k_indices
                vals = col_values[node_id]
                mu = mean(vals)
                σ = std(vals)
                cutoff = max(abs(mu) + std_multiplier * σ, absolute_min)
                col_cutoff[node_id] = cutoff
            end
        end

        # Now build I,J,V using the per-column cutoffs
        for i in 1:N_samples
            Y[i] = round(Int, all_sims[i].outcomes[target_idx])
            for (col_idx, node_id) in enumerate(top_k_indices)
                val = col_values[node_id][i]
                if abs(val) >= col_cutoff[node_id]
                    push!(I, i)
                    push!(J, col_idx)
                    push!(V, val)
                end
            end
        end

    elseif mode == :per_sample_topk
        # For every sample, keep only the top-M nodes (by abs activity) among the selected nodes.
        for (i, sim) in enumerate(all_sims)
            Y[i] = round(Int, sim.outcomes[target_idx])

            # Create list of (node_id, absval, rawval) restricted to top_k_indices
            pairs = Vector{Tuple{Int,Float64,Float64}}()
            for (col_idx, node_id) in enumerate(top_k_indices)
                val = sim.p[node_id]
                push!(pairs, (col_idx, abs(val), val))
            end

            # sort by absval descending and take up to topk_per_sample
            sort!(pairs, by = x -> x[2], rev=true)
            m = min(length(pairs), topk_per_sample)
            for j in 1:m
                col_idx, absval, rawval = pairs[j]
                if absval > absolute_min
                    push!(I, i)
                    push!(J, col_idx)
                    push!(V, rawval)
                end
            end
        end

    else
        error("Unsupported mode: $mode")
    end

    # 3) Build sparse matrix
    X_sparse_raw = sparse(I, J, V, N_samples, K_actual)

    # 4) Compute sparsity and print
    total_entries = N_samples * K_actual
    nnz_count = nnz(X_sparse_raw)
    sparsity_percent = round(1.0 - nnz_count / total_entries, digits=4) * 100

    println("  -> Data Dimensions for K=$K_actual: $N_samples samples x $K_actual features")
    println("  -> Nonzeros (nnz): $nnz_count / $total_entries ; Sparsity (approx): $sparsity_percent%")

    # 5) also compute unique exceptional nodes set for later comparison
    all_sim_exceptional_indices = vcat([s.top_entropy.idx for s in all_sims if hasproperty(s, :top_entropy)]...)
    sim_exceptional_nodes = Set(all_sim_exceptional_indices)

    return X_sparse_raw, Y, K_actual, sim_exceptional_nodes, top_k_indices, idx_to_col
end



# --- Proximal Algorithms REGRESSION (GLMNet) ---
# ----------------------------
# Sparse Logistic LASSO Solver
# ----------------------------
function sparse_logistic_lasso(X::SparseMatrixCSC, Y::Vector{Int};
    λ::Float64=0.01, maxiter::Int=500)
    N, K = size(X)
    Yf = Float64.(Y)

    # Logistic loss
    f(w) = sum(log1p(exp(-((X*w) .* (2Yf .- 1))))) / N

    # L1 regularization
    g = ProximalAlgorithms.L1Norm(λ)

    w0 = zeros(K)

    # Proximal gradient descent
    try
        β = ProximalAlgorithms.pgd(f, g, w0; maxiters=maxiter, verbose=false)
        return β
    catch e
        @warn "PGD solver failed: $e"
        return nothing
    end
end

"""
sparse_logistic_lasso(X, y; lambda=0.1, maxiter=500)

X: SparseMatrixCSC (N_samples x K_features)
y: Vector{Int} 0/1
"""
function sparse_logistic_lasso_cv(X::SparseMatrixCSC, Y::Vector{Int};
    λ_grid=10 .^ range(-4, 0, length=10),
    nfolds=5,
    maxiter=500,
    seed=42)
    Random.seed!(seed)
    N = length(Y)
    indices = collect(1:N)
    fold_sizes = ceil.(Int, N / nfolds) * ones(Int, nfolds)
    fold_sizes[end] = N - sum(fold_sizes[1:end-1])  # adjust last fold

    # Store log-loss per λ
    logloss_per_lambda = Dict{Float64, Float64}()

    for λ in λ_grid
        fold_losses = Float64[]
        for fold in 1:nfolds
            # Split indices
            test_idx = fold: nfolds: N
            train_idx = setdiff(indices, test_idx)

            X_train = X[train_idx, :]
            Y_train = Y[train_idx]
            X_test  = X[test_idx, :]
            Y_test  = Y[test_idx]

            # Fit sparse logistic LASSO
            beta = sparse_logistic_lasso(X_train, Y_train; λ=λ, maxiter=maxiter)

            # Predict probabilities on test set
            linear_pred = X_test * beta
            p_pred = 1 ./ (1 .+ exp.(-linear_pred))

            # Binary log-loss
            eps = 1e-15
            p_pred = clamp.(p_pred, eps, 1-eps)
            loss = -mean(Y_test .* log.(p_pred) .+ (1 .- Y_test) .* log.(1 .- p_pred))
            push!(fold_losses, loss)
        end
        logloss_per_lambda[λ] = mean(fold_losses)
    end

    # Pick λ with minimal log-loss
    best_lambda = argmin(logloss_per_lambda)
    best_beta = sparse_logistic_lasso(X, Y; λ=best_lambda, maxiter=maxiter)

    return best_lambda, best_beta, logloss_per_lambda
end


# --- SPARSE LASSO REGRESSION (GLMNet) ---

"""
fit_sparse_lasso_glmnet(
    X_raw,          # sparse or dense matrix (N × K)
    Y,              # 0/1 target vector
    alpha_val;      # 1.0 = LASSO, 0.0 = Ridge
    test_size=0.2,
    rng_seed=42,
    lambda_grid=nothing,
    jitter_std=1e-6
)

Returns:
    (r2_mcfadden, beta, intercept, best_lambda, test_logloss)
"""
function fit_sparse_lasso_glmnet(
    X_raw::AbstractMatrix,
    Y::Vector{Int},
    alpha_val::Float64;
    test_size::Float64 = 0.2,
    rng_seed::Int = 42,
    lambda_grid = nothing,
    jitter_std::Float64 = 1e-6
)
    # ==== 1. Data Split ========================================================
    N = length(Y)
    Random.seed!(rng_seed)

    n_test = max(1, floor(Int, N * test_size))
    test_idx = sample(1:N, n_test; replace=false)
    train_idx = setdiff(1:N, test_idx)

    X_train = X_raw[train_idx, :]
    X_test  = X_raw[test_idx, :]
    Y_train = Y[train_idx]
    Y_test  = Y[test_idx]

    @info "Train $(length(Y_train)), Test $(length(Y_test)). Positives in train: $(sum(Y_train))"

    # ==== 2. Standardization (train-only stats, no leakage) ====================
    X_train = Matrix(X_train)  # convert to dense for glmnet (fastest)
    X_test  = Matrix(X_test)

    K = size(X_train, 2)

    μ = mean(X_train, dims=1)
    σ = std(X_train, dims=1)

    # avoid zero std
    σ[σ .< 1e-12] .= 1.0

    # apply standardization
    X_train .= (X_train .- μ) ./ σ
    X_test  .= (X_test .- μ) ./ σ

    # ==== 3. Jitter injection ==================================================
    X_train .+= jitter_std .* randn(size(X_train))

    # ==== 4. Lambda grid =======================================================
    if lambda_grid === nothing
        lambda_grid = 10 .^ range(0.0, -4.0, length=30)
    end

    # ==== 5. Cross-validated logistic LASSO ====================================
    cv = glmnetcv(
        X_train, Y_train;
        nfolds = 5,
        family = "binomial",
        alpha = alpha_val,
        lambda = lambda_grid,
        standardize = false,
    )

    best_lambda = cv.lambda_min

    # ==== 6. Refit at best lambda =============================================
    fit = glmnet(
        X_train, Y_train;
        family = "binomial",
        alpha = alpha_val,
        lambda = [best_lambda],
        standardize = false,
    )

    β = vec(fit.beta[:, 1])     # coefficients
    b0 = fit.a0[1]              # intercept

    # ==== 7. Evaluate on test set =============================================
    logits = X_test * β .+ b0
    p = 1 ./(1 .+ exp.(-logits))

    # numerical stability
    eps = 1e-15
    p = clamp.(p, eps, 1 - eps)

    # log-loss
    test_logloss = -mean(Y_test .* log.(p) .+ (1 .- Y_test) .* log.(1 .- p))

    # null model loss
    pbar = mean(Y_train)
    L0 = -mean(Y_test .* log.(pbar) .+ (1 .- Y_test) .* log.(1 .- pbar))

    r2_mcfadden = L0 == 0 ? 0.0 : 1 - test_logloss / L0

    return r2_mcfadden, β, b0, best_lambda, test_logloss
end
#=





"""
fit_sparse_lasso_glmnet(X_raw::SparseMatrixCSC, Y::Vector{Int}, alpha_val::Float64;
                        test_size::Float64=0.2, rng_seed::Int=42, lambda_grid=nothing,
                        jitter_std::Float64 = 1e-6)

Performs ElasticNet logistic regression using GLMNet.jl with CV over `lambda_grid`.
- X_raw: N x K sparse matrix (samples x features)
- Y: vector of 0/1 integers
- alpha_val: ElasticNet mixing (1.0 => LASSO)
Returns: (r2_mcfadden, beta, intercept, best_lambda, test_logloss)
"""
function fit_sparse_lasso_glmnet(
    X_raw::SparseMatrixCSC,
    Y::Vector{Int},
    alpha_val::Float64;
    test_size::Float64 = 0.2,
    rng_seed::Int = 42,
    lambda_grid = nothing,
    jitter_std::Float64 = 1e-6
)
    # ---------- Split ----------
    println("TOTAL samples: ", length(Y))
    println("Positive (1):  ", sum(Y))
    println("Negative (0):  ", length(Y) - sum(Y))
    println("Positives %:   ", round(sum(Y) / length(Y) * 100, digits=4), "%")
    println("Unique(Y):      ", unique(Y))

    N = length(Y)
    Random.seed!(rng_seed)
    n_test = max(1, floor(Int, N * test_size))
    test_idx = sample(1:N, n_test; replace=false)
    train_idx = setdiff(1:N, test_idx)
    
    X_train = X_raw[train_idx, :]
    X_test  = X_raw[test_idx, :]
    Y_train = Y[train_idx]
    Y_test  = Y[test_idx]
    
    @info "Train size: $(length(Y_train)), Test size: $(length(Y_test)) ; Positives in train: $(sum(Y_train))"
    
    # ---------- Column means/std from TRAIN only (works with sparse) ----------
    # Compute feature-wise means and stds robustly (for sparse matrices)
    # mean_j = sum(X[:,j]) / n_train
    n_train = size(X_train, 1)
    K = size(X_train, 2)
    
    feature_means = zeros(Float64, K)
    feature_stds  = zeros(Float64, K)
    
    # Efficient column sums for sparse matrix:
    col_sums = vec(sum(X_train, dims=1))
    feature_means .= col_sums ./ n_train
    
    # For standard deviation: compute E[x^2] - mean^2
    col_sq_sums = vec(sum(X_train .^ 2, dims=1))
    feature_vars = col_sq_sums ./ n_train .- feature_means.^2
    # numerical noise floor
    feature_vars .= max.(feature_vars, 0.0)
    feature_stds .= sqrt.(feature_vars)
    
    # Replace near-zero stds with 1.0 to produce zero-centered columns (so glmnet doesn't blow)
    # and keep them zero after standardization
    small = feature_stds .< 1e-12
    feature_stds[small] .= 1.0
    
    # ---------- Standardize (sparse -> dense or keep sparse) ----------
    # GLMNet accepts dense or sparse; but easiest is to produce dense Matrix for glmnet.
    # If K or N is huge, you can standardize in-place or use sparse-compatible strategies.
    X_train_dense = Array(X_train)  # convert to dense; change if memory-constrained
    X_test_dense  = Array(X_test)
    
    for j in 1:K
        μ = feature_means[j]
        σ = feature_stds[j]
        X_train_dense[:, j] .= (X_train_dense[:, j] .- μ) ./ σ
        X_test_dense[:, j]  .= (X_test_dense[:, j]  .- μ) ./ σ
    end
    
    # ---------- Jitter to break separation ----------
    X_train_dense .+= jitter_std .* randn(size(X_train_dense))
    
    # ---------- Lambda grid ----------
    if lambda_grid === nothing
        log_lambda_max = 0.0
        log_lambda_min = -4.0
        num_lambdas = 30
        lambda_grid = 10 .^ range(log_lambda_max, log_lambda_min, length=num_lambdas)
    end
    
    pos = sum(Y_train)
    neg = length(Y_train) - pos
    weight_pos = neg / max(pos, 1)

    w = ones(Float64, length(Y_train))
    w[Y_train .== 1] .= weight_pos
    # ---------- Run glmnetcv (binomial family) ----------
    cv = glmnetcv(X_train_dense, Y_train;
                  nfolds=5,
                  family="binomial",
                  alpha=alpha_val,
                  lambda=lambda_grid,
                  standardize=false)  # we already standardized
    best_lambda = cv.lambda_min  # lambda that minimized CV error
    
    # Refit at best lambda to get coefficients (glmnet returns path; use glmnet with chosen lambda)
    fit = glmnet(X_train_dense, Y_train; family="binomial", alpha=alpha_val, lambda=[best_lambda], standardize=false)
    
    # Extract coefficients: fit.beta is K x nlambda, fit.a0 is intercept vector
    beta_standardized = vec(fit.beta[:, 1])   # length K
    intercept = fit.a0[1]
    
    # ---------- Evaluate on test set ----------
    # Compute probabilities using logistic: p = 1 / (1 + exp(-(X * beta + intercept)))
    logits = X_test_dense * beta_standardized .+ intercept
    proba = 1 ./(1 .+ exp.(-logits))
    
    # Clip for log-loss
    eps = 1e-15
    proba = clamp.(proba, eps, 1 - eps)
    # Binary cross-entropy
    test_logloss = -mean(Y_test .* log.(proba) .+ (1 .- Y_test) .* log.(1 .- proba))
    
    # Null model loss (predicting pbar from train)
    pbar = mean(Y_train)
    L0 = -mean(Y_test .* log.(pbar) .+ (1 .- Y_test) .* log.(1 - pbar))
    r2_mcfadden = L0 == 0.0 ? 0.0 : 1.0 - (test_logloss / L0)
    
    return r2_mcfadden, beta_standardized, intercept, best_lambda, test_logloss
end
=#




"""
Performs Elastic Net/Lasso regression using GLMNet on the sparse matrix.
"""
function fit_sparse_lasso(X_raw::SparseMatrixCSC, Y::Vector{Int}, alpha_val::Float64, target_outcome::Symbol, test_size::Float64=0.2)
    
    # 1. Data Splitting (MUST BE FIRST)
    N = length(Y)
    # Ensure reproducibility of the split
    Random.seed!(42) 
    test_indices = sample(1:N, floor(Int, N * test_size), replace=false)
    train_indices = setdiff(1:N, test_indices)

    X_train_raw = X_raw[train_indices, :]
    Y_train = Y[train_indices]
    X_test_raw = X_raw[test_indices, :]
    Y_test = Y[test_indices]
    
    @info "Training set split for $target_outcome: Ones=$(sum(Y_train)), Zeros=$(length(Y_train) - sum(Y_train))" 
    
    # 2. Leakage-Free Standardization and Jitter Injection 🚀 CRITICAL FIX
    
    # Convert sparse raw data to dense for column-wise statistics
    X_train_dense = Matrix(X_train_raw)
    X_test_dense = Matrix(X_test_raw)

    K_actual = size(X_train_dense, 2)
    
    # Compute mean and standard deviation ONLY on the training data.
    feature_means = mean(X_train_dense, dims=1)
    feature_stds = std(X_train_dense, dims=1)
    
    # Apply standardization: (X - mu) / sigma
    X_train_standardized = similar(X_train_dense)
    X_test_standardized = similar(X_test_dense)
    
    for j in 1:K_actual
        mu = feature_means[j]
        sigma = feature_stds[j]
        
        if sigma > 1e-9
            # Standardize training data
            X_train_standardized[:, j] = (X_train_dense[:, j] .- mu) ./ sigma
            # Apply same mu and sigma to test data (No leakage)
            X_test_standardized[:, j] = (X_test_dense[:, j] .- mu) ./ sigma
        else
            # Set constant features to zero
            X_train_standardized[:, j] = zeros(size(X_train_dense, 1))
            X_test_standardized[:, j] = zeros(size(X_test_dense, 1))
        end
    end

    # 🚨 JITTER INJECTION: Break Perfect Separation 🚨
    # Add a tiny amount of Gaussian noise to the standardized training features
    JITTER_STD = 1e-6
    X_train_standardized .+= JITTER_STD .* randn(size(X_train_standardized))

    # 3. Convert Standardized Data to MLJ format (Table and Categorical)
    X_mlj = MLJ.table(X_train_standardized)
    X_test_mlj = MLJ.table(X_test_standardized)
    
    # Target variable must be categorical for Logistic Regression
    Y_mlj = categorical(Y_train, levels=[0, 1]) 
    Y_test_mlj = categorical(Y_test, levels=[0, 1])

    # 4. Define Log-Loss Helper Function 
    function log_loss(Y_actual, Y_proba)
        # Convert categorical Y_actual back to binary (0/1) for log loss formula
        Y_binary = MLJ.int(Y_actual) .- 1 
        Y_proba_clipped = clamp.(Y_proba, 1e-15, 1 - 1e-15)
        # Calculate Binary Cross-Entropy (Log Loss)
        return -mean(Y_binary .* log.(Y_proba_clipped) .+ (1 .- Y_binary) .* log.(1 .- Y_proba_clipped))
    end
    
    # 5. Define Lambda Grid and Model Type
    log_lambda_min = -4.0
    log_lambda_max = 0.0
    num_lambdas = 30
    lambda_grid = exp10.(range(log_lambda_max, stop=log_lambda_min, length=num_lambdas))
    
    ModelType = @load ElasticNetRegressor pkg=MLJLinearModels verbosity=0 
    
    # Instantiate the base model with the fixed alpha (gamma)
    base_model = ModelType(
        gamma = alpha_val,  # The Elastic Net mixing ratio (alpha)
        fit_intercept = true
    )
    
    # 6. Define Tuning Strategy (Cross-Validation over the lambda grid)
    
    r = range(base_model, :lambda, values=lambda_grid)
    
    tuned_model = TunedModel(
        model = base_model,
        ranges = r,
        resampling = CV(nfolds=5), # 5-fold cross-validation
        measure = log_loss, 
        verbosity = 2, 
        train_best = true 
    )

    @info "Attempting MLJ fit using 5-Fold Cross-Validation over $(num_lambdas)  values."
    
    # 7. Train the model (Runs the CV and selects the best lambda)
    machine = MLJ.fit!(machine(tuned_model, X_mlj, Y_mlj))
    
    if MLJ.report(machine).success === false
        @error "MLJ TunedModel fit failed. Returning NaN results."
        return NaN, zeros(K_actual), NaN 
    end
    
    # 8. Extract Coefficients and Evaluate Performance
    
    # Get the best model found by CV
    best_model_machine = MLJ.fitted_params(machine).best_fit_result 
    
    final_beta_standardized = best_model_machine.fitresult.coef 
    intercept = best_model_machine.fitresult.intercept 

    # Predict probabilities on the test set
    Y_pred_dist = MLJ.predict(machine, X_test_mlj) 
    Y_pred_proba = [MLJ.pdf(d, 1) for d in Y_pred_dist] # Probability of class 1
    
    # Calculate Log-Loss of the fitted model (LM)
    LM = log_loss(Y_test_mlj, Y_pred_proba) 
    
    # Calculate Null Model Loss (L0) using the training set class imbalance
    p_bar = mean(MLJ.int(Y_train) .- 1)
    L0 = log_loss(Y_test_mlj, fill(p_bar, length(Y_test_mlj)))
    
    # Calculate McFadden's Pseudo R-squared
    r_squared_mcfadden = L0 == 0.0 ? 0.0 : 1.0 - (LM / L0)
    
    best_lambda_found = MLJ.report(machine).best_model.lambda
    
    @info "MLJ Fit Success. Best  found: $best_lambda_found. McFadden's R-squared: $r_squared_mcfadden"
    
    return r_squared_mcfadden, final_beta_standardized, intercept
end

# --- ANALYSIS AND MAPPING ---

"""
Ranks the non-zero coefficients and selects the absolute Top N.
"""
function select_top_blame_nodes(beta::AbstractVector{<:Real}, top_n::Int, top_k_indices::Vector{Int})
    # Create tuples of (coefficient_magnitude, node_index_in_original_network, original_coefficient)
    ranked_nodes = [(abs(beta[i]), top_k_indices[i], beta[i]) for i in 1:length(beta)]
    
    # Filter for non-zero/significant coefficients
    filter!(t -> t[1] > 1e-6, ranked_nodes)
    
    # Sort in descending order of magnitude
    sort!(ranked_nodes, rev=true, by = x -> x[1])

    # Select the Top N
    num_to_select = min(top_n, length(ranked_nodes))
    top_nodes = ranked_nodes[1:num_to_select]

    # Prepare output DataFrame
    df = DataFrame(
        Rank = 1:length(top_nodes),
        NodeIndex = [t[2] for t in top_nodes],
        BlameCoefficient = [t[3] for t in top_nodes],
        AbsoluteBlame = [t[1] for t in top_nodes]
    )
    
    return df
end

"""
Maps the final coefficients back to the original Node IDs for detailed printing.
"""
function get_predicted_nodes_with_coefficients(
    coefficients::AbstractVector{<:Real}, 
    top_k_indices::Vector{Int}
)
    predicted_nodes_list = []
    
    for (i, coeff) in enumerate(coefficients)
        if abs(coeff) > 1e-6 # Only include nodes selected by Lasso (non-zero coeff)
            node_id = top_k_indices[i]
            push!(predicted_nodes_list, (
                Node_ID = node_id,
                P_coeff = coeff # P_coeff is the single feature coefficient
            ))
        end
    end
            
    return sort(predicted_nodes_list, by = x -> x.Node_ID)
end

function load_node_metadata(csv_path::String)
    println("\nLoading Node Metadata from $csv_path...")
    expected_headers = [
        :node_id, :idx, :x, :y, :z, :probability, :target_pi, 
        :local_entropy_bits, :log_prob, :degree, :region # <-- Ensures this symbol is used
    ]
    # Load the CSV file
    df = CSV.read(csv_path, DataFrame,header=expected_headers,delim=',')
    actual_names = names(df)

    # This print statement is CRITICAL for diagnosing the name mismatch
    #println("--- ACTUAL COLUMN NAMES READ FROM CSV ---")
    #println(actual_names)
    #println("-----------------------------------------")

    # Now, check for the column name. If it fails here, the manual file fix (Step 1) failed.
    # 2. Check if the exact string "region" is present
    if !("region" in actual_names) 
        # Since we've confirmed the name is there, this should no longer execute.
        # If it does, the column is genuinely missing or renamed internally later.
        @error "CRITICAL ERROR: 'region' column not found in CSV..."
        num_rows = nrow(df)
        df[!, :RegionName] = fill("No Region Data", num_rows)
        # We must return a modified dataframe, so we cannot return early.
        region_col_sym = :Missing
    else
        # 3. If found, use the known symbol to access the column
        region_col_sym = :region
        
        # 4. Extract the region name from the string format "['REGION']"
        df[!, :RegionName] = map(df[!, region_col_sym]) do r
            m = match(r"\[\s*\'(.*?)\'\s*\]", r)
            return m !== nothing ? m.captures[1] : "Unknown"
        end
    end

    # 2. Rename the primary key columns
    rename!(df, :node_id => :NodeID_Original, :idx => :NodeIndex)
    # Keep only the essential columns for mapping
    metadata_df = df[:, [:NodeIndex, :NodeID_Original, :RegionName, :degree, :x, :y, :z]]

    println("  -> Successfully loaded $(nrow(metadata_df)) nodes.")
    
    return metadata_df
end

"""
Compares the K selected nodes vs the final Lasso-predicted nodes and prints the locus.
"""
function compare_predicted_vs_selected_nodes(
    predicted_nodes_df::DataFrame, 
    simulated_locus_nodes::Set{Int},
    K_final::Int,
    target_outcome::Symbol,
    metadata_df::DataFrame # <-- NEW ARGUMENT
)
    # 1. Prepare the Locus Sets
    predicted_locus_set = Set(predicted_nodes_df.NodeIndex)
    sim_locus_set = simulated_locus_nodes
    
    # 2. Calculate Overlap
    intersection_nodes = intersect(predicted_locus_set, sim_locus_set)
    overlap_count = length(intersection_nodes)
    
    # ----------------------------------------------------------------------
    # 3. CALCULATE METRICS (FIX: Must be defined before printing in step 4)
    # ----------------------------------------------------------------------
    predicted_size = length(predicted_locus_set)
    simulated_size = length(sim_locus_set)

    # Calculate overlap percentages (avoiding division by zero if sets are empty)
    overlap_percent_predicted = predicted_size > 0 ? (overlap_count / predicted_size) * 100 : 0.0
    overlap_percent_simulated = simulated_size > 0 ? (overlap_count / simulated_size) * 100 : 0.0

    # 4. Generate Report
    
    println("\n" * repeat("=", 70))
    println("      ✨ EXCEPTIONAL LOCUS VALIDATION REPORT: $target_outcome")
    println(repeat("=", 70))
    println("Model: Sparse Elastic Net (K=$K_final features)")
    println(repeat("-", 70))
    
    println("## Predicted Exceptional Locus (Top Lasso Coefficients)")
    # This line now safely uses the variable calculated in step 3.
    println("Total Size: $predicted_size nodes.") 

    # 5. Enrich Predicted Locus with Region Data (Corrected Join Logic)
    
    # Merge the Lasso results (predicted_nodes_df) with the metadata
    # The key is 'NodeIndex' in predicted_nodes_df, which matches 'NodeIndex' 
    # (or whatever you renamed the sequential index 'idx' to) in metadata_df.
    # IMPORTANT: Ensure your load_node_metadata function uses `rename!(df, :idx => :NodeIndex)`
    
    enriched_df = innerjoin(
        predicted_nodes_df, 
        metadata_df, 
        on = :NodeIndex, # Use single symbol if both columns have the same name (NodeIndex)
        makeunique = true
    )

    # Reorder and display the final, enriched table
    final_report_cols = [:Rank, :NodeIndex, :RegionName, :BlameCoefficient, :degree]
    display(enriched_df[:, final_report_cols])
    println(repeat("-", 70))

    # Print Simulated Locus Summary (Uses simulated_size)
    println("## Simulated Exceptional Locus (Simulation Ground Truth)")
    println("Total Size: $simulated_size nodes (Nodes flagged as highly stable/low-entropy during simulation).")
    
    # Show the first few simulated nodes as a sample
    sample_nodes = collect(sim_locus_set)[1:min(10, simulated_size)]
    println("Sample Nodes: $(sample_nodes...) $(simulated_size > 10 ? "..." : "")")
    println(repeat("-", 70))
    
    # Print Locus Overlap Summary (Uses calculated percentages)
    println("## Locus Overlap Summary")
    println("Overlap Count: $overlap_count nodes.")
    println("Overlapping Nodes: $intersection_nodes")
    
    @printf("Overlap (Predicted Locus Coverage): %.2f%% (%.0f of %.0f nodes found in Sim Locus)\n", 
             overlap_percent_predicted, 
             Float64(overlap_count), 
             Float64(predicted_size)
    )
    
    @printf("Overlap (Simulated Locus Coverage): %.2f%% (%.0f of %.0f nodes explained by Predicted Locus)\n", 
             overlap_percent_simulated, 
             Float64(overlap_count), 
             Float64(simulated_size)
    )
    println(repeat("=", 70))

    # Print the specific blame nodes that overlapped, now with region info
    if overlap_count > 0
        println("### Overlapping Nodes Details (Predicted Rank, Enriched)")
        
        # Filter the enriched DataFrame
        overlap_df = filter(row -> row.NodeIndex in intersection_nodes, enriched_df)
        display(overlap_df[:, final_report_cols])
        println(repeat("-", 70))
    end
end

# --- TOP-LEVEL PIPELINE (MAIN EXECUTION) 8 time run for joining nodes.---
# One vs Rest solution -- multilabel classification
# from analyze_bson.jl
# Dict(:blurred_vision => 0.051702298916927375, 
#      :coma => 0.2568765724418855, 
#      :confusion => 0.25392813537065895, 
#      :epilepsy => 0.05543901822354224, 
#      :sweating => 0.04931152769765666, 
#      :energized_alert => 0.03773483537055573, 
#      :hyperactivity => 0.24829003355530585, 
#      :anxiety => 0.046717578423467694), 
#      CONFIDENCE 0.7089533056577472, 
#      Actual => [0, 0, 0, 1, 1, 0, 1, 0]
function run_analysis_pipeline_ovr(data_file::String)
    
    # A. Initial Data Loading (Called once)
    
    # A-1. Load all simulation objects (N=1500 samples)
    data = BSON.load(data_file)
    all_sims = data[:sims]
    N_total_samples = length(all_sims)
    N_total_features = length(all_sims[1].p)
    
    # A-2. Load Node Metadata (Called once)
    metadata_df = load_node_metadata(METADATA_CSV_PATH)
    
    println("Analysis Initialized:")
    println("  Total Samples: $N_total_samples")
    println("  Total Raw Features (Nodes): $N_total_features")
    println("  Node Metadata Loaded: $(nrow(metadata_df)) entries")
    println(repeat("=", 70))
    
    all_outcome_results = Dict{Symbol, NamedTuple}()

    # 🚀 OUTER LOOP: Iterate over all 8 outcomes (One-vs-Rest)
    for target_sym in OUTCOMES
        
        current_target_outcome = target_sym
        
        println("\n=======================================================")
        println(">>> RUNNING REVERSE MAP FOR TARGET: $current_target_outcome <<<")
        println("=======================================================")
        
        # FIX: Initialize variables for the current outcome run HERE
        last_beta = Float64[]
        last_K_final = 0
        sim_exceptional_nodes = Set{Int}()
        top_k_indices_final = Int[]
        
        println("\n1. Starting Feature Set Optimization (N=$N_total_features -> K features)...")
        println(repeat("-", 50))
        println("| Feature Count (K) | Samples (N) | R-squared |")
        println(repeat("-", 50))

        # B. Optimization Loop: Test performance vs. feature set size (K)
        # -----------------------------------------------------------------
        for k_final in FEATURE_COUNT_CANDIDATES # <-- THIS WAS THE MISSING LOOP
            
            # 2. Prepare Sparse Data (Filtering N -> K, then Sparse Matrix Construction)
            X, Y, K_actual, current_sim_exceptional_nodes, top_k_indices, _ = prepare_sparse_regression_data(
                all_sims, current_target_outcome, OUTCOMES, k_final; mode=:per_feature_percentile, percentile_keep=0.90
            )
            
            # 3. Fit Sparse Lasso and Evaluate
            try
                #r_squared, final_beta = fit_sparse_lasso_glmnet(Matrix(X), Y, LASSO_ALPHA, current_target_outcome) # Assuming returns r_squared, final_beta
                #r_squared, final_beta = fit_sparse_lasso_glmnet(Matrix(X), Y, LASSO_ALPHA)
                
                # Define λ grid for CV (tune this as needed)
                λ_grid = 10 .^ range(-4, -1, length=5)  # e.g., [1e-4, 1e-3, 1e-2, 1e-1]

                best_lambda, final_beta, logloss_per_lambda = sparse_logistic_lasso_cv(X, Y; λ_grid=λ_grid)

                # Optional: compute pseudo R² (McFadden)
                p_bar = mean(Y)
                L0 = -mean(Y .* log.(p_bar) .+ (1 .- Y) .* log.(1 - p_bar))
                LM = -mean(Y .* log.(1 ./ (1 .+ exp.(-X * final_beta))) .+ (1 .- Y) .* log.(1 .- 1 ./ (1 .+ exp.(-X * final_beta))))
                r_squared = 1 - (LM / L0)

                @printf("| %-17d | %-11d | %.4f    |\n", K_actual, N_total_samples, r_squared)
                
                if k_final == TARGET_K_FOR_DETAIL
                    # Store results for the final detailed comparison
                    last_beta = final_beta
                    last_K_final = K_actual
                    top_k_indices_final = top_k_indices
                    sim_exceptional_nodes = current_sim_exceptional_nodes # Store the set once
                end
            catch e
                @printf("| %-17d | %-11d | %-8s |\n", K_actual, N_total_samples, "-Error")
                # Handle error if K=TARGET_K_FOR_DETAIL failed
                #if k_final == TARGET_K_FOR_DETAIL
                #    last_beta = Float64[]
                #end
                # Do NOT update last_beta or last_K_final here. They should remain 
                # at their last successful value or their initial state (0, []).
                
                # Optional: Print the error itself for detailed debugging
                # @warn "Solver failed for K=$k_final: $e"
            end
        end
        # -----------------------------------------------------------------
        
        println(repeat("-", 50))
        println("\n4. Optimization Complete for $current_target_outcome.")

        # C. Run Detailed Prediction Analysis for TARGET_K_FOR_DETAIL
        if last_K_final > 0 && !isempty(last_beta)
            
            # 1. Select the top N nodes based on the optimal K's coefficients
            top_blame_nodes_df = select_top_blame_nodes(last_beta, TOP_N_NODES, top_k_indices_final)
            
            # 2. Run Comparison and Report Generation (Integration Point)
            compare_predicted_vs_selected_nodes(
                top_blame_nodes_df, 
                sim_exceptional_nodes,
                last_K_final,
                current_target_outcome,
                metadata_df
            )

            # 3. Store the final result for this outcome
            all_outcome_results[current_target_outcome] = (
                top_nodes_df = top_blame_nodes_df,
                locus_size = length(top_blame_nodes_df.NodeIndex),
                K_final = last_K_final,
                sim_locus = sim_exceptional_nodes
            )

        else
            println("\nSkipping detailed prediction analysis: TARGET_K_FOR_DETAIL ($TARGET_K_FOR_DETAIL) failed for $current_target_outcome.")
        end
    end # End of OUTER LOOP

    println("\n" * repeat("=", 70))
    println("All 8 Outcomes Analyzed. Final results stored in 'all_outcome_results'.")
    println(repeat("=", 70))
    
    return all_outcome_results
end



# --- TOP-LEVEL PIPELINE (MAIN EXECUTION) ---

function run_analysis_pipeline()
    
    # A. Load all simulation objects once
    data = BSON.load(DATA_FILE)
    all_sims = data[:sims]
    N_total_samples = length(all_sims)
    N_total_features = length(all_sims[1].p)
    
    println("Analysis Initialized:")
    println("  Total Samples: $N_total_samples")
    println("  Total Raw Features (Nodes): $N_total_features")
    println(repeat("-", 70))
    
    println("\n1. Starting Feature Set Optimization (N=$N_total_features -> K features)...")
    println("--------------------------------------------------")
    println("| Feature Count (K) | Samples (N) | R-squared |")
    println("--------------------------------------------------")

    last_beta = Float64[]
    last_K_final = 0
    sim_exceptional_nodes = Set{Int}()
    top_k_indices_final = Int[]

    # B. Optimization Loop: Test performance vs. feature set size (K)
    for k_final in FEATURE_COUNT_CANDIDATES
        
        # 2. Prepare Sparse Data (Filtering N -> K, then Sparse Matrix Construction)
        X, Y, K_actual, sim_exceptional_nodes, top_k_indices, _ = prepare_sparse_regression_data(
            all_sims, TARGET_OUTCOME, OUTCOMES, k_final; mode=:per_feature_percentile, percentile_keep=0.90
        )
        
        # 3. Fit Sparse Lasso and Evaluate
        #r_squared, final_beta, _ = fit_sparse_lasso_glmnet(X, Y, LASSO_ALPHA,TARGET_OUTCOME)
        #r_squared, final_beta, _ = fit_sparse_lasso_glmnet(Matrix(X), Y, LASSO_ALPHA)
        # Define λ grid for CV (tune this as needed)
        λ_grid = 10 .^ range(-4, -1, length=5)  # e.g., [1e-4, 1e-3, 1e-2, 1e-1]

        best_lambda, final_beta, logloss_per_lambda = sparse_logistic_lasso_cv(X, Y; λ_grid=λ_grid)

        # Optional: compute pseudo R² (McFadden)
        p_bar = mean(Y)
        L0 = -mean(Y .* log.(p_bar) .+ (1 .- Y) .* log.(1 - p_bar))
        LM = -mean(Y .* log.(1 ./ (1 .+ exp.(-X * final_beta))) .+ (1 .- Y) .* log.(1 .- 1 ./ (1 .+ exp.(-X * final_beta))))
        r_squared = 1 - (LM / L0)

        #@printf("| %-17d | %-11d | %.4f    |\n", K_actual, N_samples, r_squared)
        
        @printf("| %-17d | %-11d | %.4f    |\n", K_actual, N_total_samples, r_squared)
        
        
        
        #println("| $(K_actual:<17) | $(N_total_samples:<11) | $(@sprintf("%.4f", r_squared))    |")

        if k_final == TARGET_K_FOR_DETAIL
             # Store results for the final detailed comparison
             last_beta = final_beta
             last_K_final = K_actual
             top_k_indices_final = top_k_indices
        end
    end

    println("--------------------------------------------------")
    println("\n4. Optimization Complete.")
    
    # C. Run Detailed Prediction Analysis for TARGET_K_FOR_DETAIL
    if last_K_final > 0
        
        # Select the top N nodes based on the optimal K's coefficients
        top_blame_nodes_df = select_top_blame_nodes(last_beta, TOP_N_NODES, top_k_indices_final)
        
        # Compare and print the final exceptional locus
        compare_predicted_vs_selected_nodes(
            top_blame_nodes_df, 
            sim_exceptional_nodes,
            last_K_final
        )
    else
        println("\nSkipping detailed prediction analysis: TARGET_K_FOR_DETAIL ($TARGET_K_FOR_DETAIL) was not processed.")
    end
end

#run_analysis_pipeline_ovr(DATA_FILE)
using SparseArrays, Random, Statistics

# ==========================================
# Sparse Reverse Map Pipeline
# ==========================================
function run_sparse_pipeline_2(all_sims::Vector{Any}, 
    outcomes::Vector{Symbol}, 
    candidate_K::Vector{Int}, 
    target_K_for_detail::Int=500, 
    top_n_nodes::Int=10)

N_total_samples = length(all_sims)
N_total_features = length(all_sims[1].p)
println("Analysis Initialized: $N_total_samples samples, $N_total_features features.")

all_outcome_results = Dict{Symbol, NamedTuple}()

for target_sym in outcomes
println("\n=======================================================")
println(">>> RUNNING SPARSE REVERSE MAP FOR TARGET: $target_sym <<<")
println("=======================================================")

last_beta = nothing
last_K_final = 0
top_k_indices_final = Int[]

println("| Feature Count (K) | Samples (N) | R-squared |")
println(repeat("-", 50))

for k_final in candidate_K
try
# -------------------------
# 1. Prepare raw sparse matrix
# -------------------------
X_sparse, Y, _, sim_exceptional_nodes, _, _ = 
prepare_sparse_regression_data(all_sims, target_sym, outcomes, k_final)

# -------------------------
# 2. Select top-K by column signal (sum of absolute values)
# -------------------------
K_actual = min(k_final, size(X_sparse, 2))
col_scores = vec(sum(abs.(X_sparse), dims=1))  # flatten to 1D
top_idx = sortperm(col_scores, rev=true)[1:K_actual]
X_final = X_sparse[:, top_idx]
final_node_indices = top_idx

# -------------------------
# 3. Add tiny jitter to avoid zero columns
# -------------------------
X_final .= X_final .+ 1e-6 .* randn(size(X_final))

# -------------------------
# 4. Fit sparse logistic LASSO
# -------------------------
λ_grid = 10 .^ range(-4, -1, length=5)
best_lambda, final_beta, loss = sparse_logistic_lasso_cv(X_final, Y; λ_grid=λ_grid)

if final_beta === nothing
@printf("| %-17d | %-11d | %-8s |\n", K_actual, N_total_samples, "-SolverFail")
continue
end

# -------------------------
# 5. Compute McFadden pseudo-R²
# -------------------------
Yf = Float64.(Y)
p_bar = mean(Yf)
L0 = -mean(Yf .* log.(p_bar) .+ (1.0 .- Yf) .* log.(1.0 .- p_bar))
LM = -mean(Yf .* log.(1.0 ./ (1.0 .+ exp.(-X_final * final_beta))) .+
  (1.0 .- Yf) .* log.(1.0 .- 1.0 ./ (1.0 .+ exp.(-X_final * final_beta))))
r_squared = 1 - LM / L0

@printf("| %-17d | %-11d | %.4f    |\n", K_actual, N_total_samples, r_squared)

# -------------------------
# 6. Store results for detailed K
# -------------------------
if k_final == target_K_for_detail
last_beta = final_beta
last_K_final = K_actual
top_k_indices_final = final_node_indices
end

catch e
@printf("| %-17d | %-11d | %-8s |\n", k_final, N_total_samples, "-Error")
end
end

println(repeat("-", 50))

# -------------------------
# 7. Select top nodes for detailed analysis
# -------------------------
if last_beta !== nothing
top_indices_sorted = sortperm(abs.(last_beta), rev=true)[1:min(top_n_nodes, length(last_beta))]
top_nodes_df = [(NodeIndex=top_k_indices_final[i], Coeff=last_beta[i]) for i in top_indices_sorted]

all_outcome_results[target_sym] = (
top_nodes_df = top_nodes_df,
K_final = last_K_final,
sim_locus = sim_exceptional_nodes
)

println("Top nodes for $target_sym: ", [n.NodeIndex for n in top_nodes_df])
else
println("Skipping detailed analysis for $target_sym (solver failed).")
end
end

return all_outcome_results
end


# ----------------------------
# Filter sparse matrix by coverage
# ----------------------------
function filter_sparse_by_coverage(X::SparseMatrixCSC, min_coverage::Float64=0.001)
    N, K = size(X)
    keep_cols = [j for j in 1:K if nnz(X[:, j]) / N >= min_coverage]
    return X[:, keep_cols], keep_cols
end

# ----------------------------
# Run sparse reverse mapping pipeline
# ----------------------------
function run_sparse_pipeline(
        all_sims::Vector,
        outcomes::Vector{Symbol},
        candidate_K::Vector{Int} = [50, 100, 200],
        target_K_for_detail::Int = 100,
        top_n_nodes::Int = 10,
        min_coverage::Float64 = 0.001
    )

    N_total_samples = length(all_sims)
    N_total_features = length(all_sims[1].p)
    println("Analysis Initialized: $N_total_samples samples, $N_total_features features.")

    all_results = Dict{Symbol, NamedTuple}()

    for target_sym in outcomes
        println("\n=======================================================")
        println(">>> RUNNING SPARSE REVERSE MAP FOR TARGET: $target_sym <<<")
        println("=======================================================")

        last_beta = nothing
        last_K_final = 0
        top_k_indices_final = Int[]
        sim_exceptional_nodes = Set{Int}()

        println("| Feature Count (K) | Samples (N) | R-squared |")
        println(repeat("-", 50))

        for k_final in candidate_K
            # -------------------------
            # 1. Prepare sparse matrix
            # -------------------------
            X_sparse, Y, _, sim_exceptional_nodes, top_k_indices, _ =
                prepare_sparse_regression_data(all_sims, target_sym, outcomes, k_final)

            # -------------------------
            # 2. Filter by coverage
            # -------------------------
            X_filtered, kept_cols = filter_sparse_by_coverage(X_sparse, min_coverage)

            K_actual = min(k_final, size(X_filtered, 2))
            if K_actual == 0
                println("Skipping K=$k_final (no columns passed coverage filter)")
                continue
            end

            # -------------------------
            # 3. Select top-K by variance
            # -------------------------
            feature_std = mapslices(std, X_filtered; dims=1)[:]  # convert 1×K to vector
            top_idx = sortperm(feature_std, rev=true)[1:K_actual]
            #top_idx = sortperm(std(X_filtered, dims=1), rev=true)[1:K_actual]
            X_final = X_filtered[:, top_idx]
            final_node_indices = kept_cols[top_idx]

            # -------------------------
            # 4. Add tiny jitter to avoid constant-columns
            # -------------------------
            X_final .= X_final .+ 1e-6 .* randn(size(X_final))

            # -------------------------
            # 5. Fit sparse LASSO (fallback Elastic Net)
            # -------------------------
            try
                λ_grid = 10 .^ range(-4, -1, length=5)
                best_lambda, final_beta, loss = sparse_logistic_lasso_cv(X_final, Y; λ_grid=λ_grid)

                if final_beta === nothing
                    # Try Elastic Net if LASSO fails
                    best_lambda, final_beta, loss = sparse_logistic_elasticnet_cv(X_final, Y; λ_grid=λ_grid, α=0.5)
                end

                if final_beta === nothing
                    @printf("| %-17d | %-11d | %-8s |\n", K_actual, N_total_samples, "-SolverFail")
                    continue
                end

                # -------------------------
                # 6. Compute McFadden pseudo-R²
                # -------------------------
                Yf = Float64.(Y)
                p_bar = mean(Yf)
                L0 = -mean(Yf .* log.(p_bar) .+ (1.0 .- Yf) .* log.(1.0 .- p_bar))
                p_pred = 1.0 ./ (1.0 .+ exp.(-X_final * final_beta))
                LM = -mean(Yf .* log.(p_pred) .+ (1.0 .- Yf) .* log.(1.0 .- p_pred))
                r_squared = 1 - LM / L0

                @printf("| %-17d | %-11d | %.4f    |\n", K_actual, N_total_samples, r_squared)

                # -------------------------
                # 7. Store detailed K
                # -------------------------
                if k_final == target_K_for_detail
                    last_beta = final_beta
                    last_K_final = K_actual
                    top_k_indices_final = final_node_indices
                end

            catch e
                @printf("| %-17d | %-11d | %-8s |\n", K_actual, N_total_samples, "-Error")
            end
        end

        println(repeat("-", 50))

        # -------------------------
        # 8. Select top nodes
        # -------------------------
        if last_beta !== nothing
            top_indices_sorted = sortperm(abs.(last_beta), rev=true)[1:min(top_n_nodes, length(last_beta))]
            top_nodes_df = [(NodeIndex=top_k_indices_final[i], Coeff=last_beta[i]) for i in top_indices_sorted]

            all_results[target_sym] = (
                top_nodes_df = top_nodes_df,
                K_final = last_K_final,
                sim_locus = sim_exceptional_nodes
            )

            println("Top nodes for $target_sym: ", [n.NodeIndex for n in top_nodes_df])
        else
            println("Skipping detailed analysis for $target_sym (solver failed).")
        end
    end

    return all_results
end

using Statistics
using LinearAlgebra
using GLM
#const OUTCOMES = [:epilepsy, 
#                    :confusion, :blurred_vision, 
#                    :sweating, :coma, 
#                    :energized_alert, 
#                    :hyperactivity, 
#                    :anxiety]
# ------------------------------------------------------------
# Prepare dense regression data from your flattened structure
# ------------------------------------------------------------
function prepare_dense_regression_data_dense(all_sims::Vector{Any}, target_idx::Int; top_k::Int=0)
    N = length(all_sims)
    K = maximum([length(s.top_entropy.idx) for s in all_sims])
    
    X = zeros(Float64, N, K)
    Y = zeros(Float64, N)
    
    for i in 1:N
        sim = all_sims[i]
        n_features = length(sim.top_entropy.idx)
        X[i, 1:n_features] .= sim.top_entropy.p_top
        Y[i] = Float64(sim.outcomes[target_idx])
    end
    
    # Remove zero-variance columns
    col_std = vec(std(X, dims=1))
    nonzero_std_idx = findall(!=(0), col_std)
    X = X[:, nonzero_std_idx]
    
    # Scale up small numbers
    X .*= 1e5
    
    # Select top-K by variance
    if top_k > 0 && size(X, 2) > top_k
        feature_std = vec(std(X, dims=1))
        top_idx = sortperm(feature_std, rev=true)[1:top_k]
        X = X[:, top_idx]
    end
    
    return X, Y
end

# ------------------------------------------------------------
# Run dense regression for multiple K candidates
# ------------------------------------------------------------
function run_dense_pipeline(all_sims::Vector{Any}, target_idx::Int, candidate_K::Vector{Int}; λ_grid=nothing)
    println(">>> RUNNING DENSE REGRESSION FOR TARGET INDEX: $target_idx <<<")
    println("--------------------------------------------------")
    println("| Feature Count (K) | Samples (N) | R-squared |")
    println("--------------------------------------------------")
    
    N = length(all_sims)
    
    for K in candidate_K
        # Prepare design matrix
        X, Y = prepare_dense_regression_data_dense(all_sims, target_idx, top_k=K)
        try
            # Fit linear regression (you can change to GLM.Logit if Y is binary)
            lm_model = lm(X, Y)
            Y_pred = predict(lm_model, X)
            r2 = 1.0 - sum((Y .- Y_pred).^2) / sum((Y .- mean(Y)).^2)
            
            @printf("| %-17d | %-11d | %.4f    |\n", K, N, r2)
        catch e
            @printf("| %-17d | %-11d | %-8s |\n", K, N, "-Error")
        end
    end
end

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

# all_sims: your loaded BSON simulations
# target_idx: which outcome column to predict (1-based)
# candidate_K: list of top-K features to test
# run_dense_pipeline(all_sims, target_idx, candidate_K)

# Example:
data = BSON.load(DATA_FILE)
all_sims = data[:sims]
target_idx = 2  # anxiety, if first in outcomes
candidate_K = [200, 500, 1000, 2000, 5000]
run_dense_pipeline(all_sims, target_idx, candidate_K)