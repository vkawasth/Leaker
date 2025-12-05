using Random, Statistics, LinearAlgebra

# ===========================
# 1. Parameters
# ===========================
n_nodes = 3_500_000        # number of nodes
n_basis = 5                # number of basis elements per node
n_steps = 5                # number of simulation steps
n_stimulus = 10            # stimuli per step
threshold = 1.0f0          # threshold for classification
top_pos = 10
top_neg = 10

rng = MersenneTwister(42)

# ===========================
# 2. Node initialization
# ===========================
# Node states: n_nodes x n_basis
states = rand(rng, Float32, n_nodes, n_basis) * 2f0 .- 1f0

# Node weights (multiplicative)
weights = rand(rng, Float32, n_nodes, n_basis) * 2f0 .- 1f0

# ===========================
# 3. Stimuli
# ===========================
stimuli = [rand(rng, Float32, n_stimulus, n_basis) * 2f0 .- 1f0 for _ in 1:n_steps]

# ===========================
# 4. Simulation
# ===========================
category_hist_thresh = Array{Int8}(undef, n_nodes, n_steps)

for t in 1:n_steps
    # Sum stimuli for this step
    step_stim = sum(stimuli[t], dims=1)        # 1 x n_basis

    # Broadcast: multiply each node state with (state + step_stim)
    states .= states .* (states .+ step_stim)  # elementwise

    # Signed norm per node
    node_responses = sum(states, dims=2)       # n_nodes x 1

    # Threshold categorization
    category_hist_thresh[:, t] .= 0
    category_hist_thresh[node_responses[:,1] .> threshold, t] .= 1
    category_hist_thresh[node_responses[:,1] .< -threshold, t] .= -1

    # Print summary
    pos = count(category_hist_thresh[:, t] .== 1)
    neg = count(category_hist_thresh[:, t] .== -1)
    neut = n_nodes - pos - neg
    println("Step $t: Positive=$pos, Negative=$neg, Neutral=$neut")
end

# ===========================
# 5. Optional: summarize a step
# ===========================
function summarize_step(states::Array{Float32,2}, category::Vector{Int8})
    positives = findall(category .== 1)
    negatives = findall(category .== -1)
    neutrals  = findall(category .== 0)

    summary = Dict(
        "positive_count" => length(positives),
        "negative_count" => length(negatives),
        "neutral_count"  => length(neutrals),
        "positive_mean"  => isempty(positives) ? 0f0 : mean(sum(states[positives, :], dims=2)),
        "negative_mean"  => isempty(negatives) ? 0f0 : mean(sum(states[negatives, :], dims=2)),
        "neutral_mean"   => isempty(neutrals)  ? 0f0 : mean(sum(states[neutrals, :], dims=2))
    )
    return summary
end
