# vbi_pipeline.jl
module VBI

using BSON, Statistics, Random, LinearAlgebra
using Flux
using MLDataPattern
using StatsBase

export build_features_from_sims,
       region_score_from_prior,
       posterior_to_region_map,
       train_posterior_py_sbi,
       train_posterior_flux_ensemble

# ---------------------------
# Helpers: region mapping
# ---------------------------
"""
build_region_index(nodes::DataFrame, region_list::Vector{String})

Returns:
 - node2region_idx :: Vector{Int} of length n_nodes mapping node_i -> region index in region_list
 - region_names :: Vector{String} (the provided region_list)
"""
function build_region_index(nodes, region_list::Vector{String})
    n = nrow(nodes)
    node2region_idx = zeros(Int, n)
    # map region name strings to their index
    rmap = Dict{String,Int}()
    for (i,r) in enumerate(region_list)
        rmap[r] = i
    end
    for i in 1:n
        rname = replace(string(nodes.regions[i]), "Region_Acronym_" => "")
        node2region_idx[i] = get(rmap, rname, 0)  # 0 = unknown region
    end
    return node2region_idx, region_list
end

# ---------------------------
# Feature builder
# ---------------------------
"""
build_features_from_sims(θs, sims; nodes, region_list)

Inputs:
 - θs: Vector of parameter structs
 - sims: Vector of simulation dicts/structs, each must have .p (final probability vector)
          and optionally .bv_metrics (Dict) and .outcomes (Vector or BitVector)
 - nodes: nodes DataFrame used by simulator (must have .regions)
 - region_list: Vector{String} of region names to aggregate over

Outputs:
 - X :: Matrix{Float32} of size (N, D) feature matrix
 - Y :: Matrix{Float32} of size (N, θdim) target parameter vectors
 - meta :: Dict with ancillary info (region_list, node2region mapping, topk lists optionally)
"""
function build_features_from_sims(θs, sims; nodes, region_list::Vector{String}, topk_nodes::Int = 0)
    N = length(sims)
    node2region_idx, region_list = build_region_index(nodes, region_list)
    m = length(region_list)

    # θ to vector conversion: convert EntropyPriorParams struct -> vector
    function θ_to_vec(θ)
        return Float32.([θ.cortex_scale, θ.hippo_scale, θ.sensory_scale, θ.cerebellum_scale, θ.noise])
    end
    θdim = length(θ_to_vec(θs[1]))

    # Feature dimension:
    # - region entropies: m
    # - region mean p: m (optional) -> include for now
    # - global H[p], min(p), KL placeholder -> 3
    # - BV metrics: assume up to 3 (n_kicks, max_corr, mean_corr) -> if missing, zeros
    # - outcomes: K (detect from sims[1].outcomes)
    K = haskey(sims[1], :outcomes) || hasfield(sims[1], :outcomes) ? length(sims[1].outcomes) : 0

    D = m + m + 3 + 3 + K
    # optionally add top-k node entropies if requested (ignored for now)
    X = zeros(Float32, N, D)
    Y = zeros(Float32, N, θdim)

    for i in 1:N
        sim = sims[i]
        p = sim[:p] isa Nothing ? sim.p : sim[:p]   # handle dict vs struct
        # compute per-node local entropy
        p_safe = max.(p, 1e-20)
        local_ent = -p_safe .* log2.(p_safe)

        # region aggregations
        region_sum_p = zeros(Float64, m)
        region_sum_ent = zeros(Float64, m)
        region_counts = zeros(Int, m)
        for (node, ridx) in enumerate(node2region_idx)
            if ridx > 0
                region_sum_p[ridx] += p[node]
                region_sum_ent[ridx] += local_ent[node]
                region_counts[ridx] += 1
            end
        end
        # avoid division by zero
        region_mean_p = [region_counts[j] > 0 ? region_sum_p[j] / region_counts[j] : 0.0 for j in 1:m]
        region_mean_ent = [region_counts[j] > 0 ? region_sum_ent[j] / region_counts[j] : 0.0 for j in 1:m]

        H_global = Float64(sum(-p_safe .* log2.(p_safe)))
        pmin = minimum(p_safe)
        # compute KL(p || π) if sim contains π or θ -> compute make_parameterized_prior using θs[i]
        kl = haskey(sim, :π) ? Float64(sum(p_safe .* log.(p_safe ./ max.(sim[:π], 1e-20)))) : 0.0

        # bv metrics
        bv_n = 0.0; bv_max = 0.0; bv_mean = 0.0
        if haskey(sim, :bv_metrics) && sim[:bv_metrics] !== nothing
            bm = sim[:bv_metrics]
            bv_n = get(bm, :n_kicks, 0.0)
            bv_max = get(bm, :max_corr, 0.0)
            bv_mean = get(bm, :mean_corr, 0.0)
        end

        # outcomes
        outvec = K == 0 ? Float32[] : Float32.(sim[:outcomes])

        # assemble feature vector
        fv = Float32[]
        append!(fv, Float32.(region_mean_ent))
        append!(fv, Float32.(region_mean_p))
        push!(fv, Float32(H_global)); push!(fv, Float32(pmin)); push!(fv, Float32(kl))
        push!(fv, Float32(bv_n)); push!(fv, Float32(bv_max)); push!(fv, Float32(bv_mean))
        append!(fv, outvec)

        X[i, :] = fv
        Y[i, :] = θ_to_vec(θs[i])
    end

    meta = Dict(:region_list => region_list, :node2region_idx => node2region_idx)
    return X, Y, meta
end


# ---------------------------
# Map θ -> region responsibility (cheap proxy)
# ---------------------------
"""
region_score_from_prior(θ, nodes, region_list)

Given a parameter vector θ (EntropyPriorParams) or θ-vector, compute region-level
prior scores π_R (normalized) and return normalized responsibility vector.
This is the cheap, analytic proxy: responsibility ∝ π_R.
"""
function region_score_from_prior(θ; nodes=nothing, region_list=nothing)
    # Expect θ is a NamedTuple or vector or the EntropyPriorParams type
    # We assume user has make_parameterized_prior(nodes, θ) available in global scope
    if nodes === nothing || region_list === nothing
        error("region_score_from_prior needs nodes and region_list as keyword args")
    end
    # call the user-provided make_parameterized_prior that exists in Main scope
    π = Main.make_parameterized_prior(nodes, θ)   # returns vector per node
    # aggregate by region_list using node2region mapping
    node2region_idx = build_region_index(nodes, region_list)[1]  # returns mapping
    m = length(region_list)
    region_mass = zeros(Float64, m)
    for i in 1:length(π)
        r = node2region_idx[i]
        if r > 0
            region_mass[r] += π[i]
        end
    end
    # normalize
    s = sum(region_mass)
    return s > 0 ? region_mass ./ s : fill(1.0/m, m)
end

# ---------------------------
# Aggregate posterior samples -> region map
# ---------------------------
"""
posterior_to_region_map(θ_samples, nodes, region_list; α=0.05)

θ_samples: Matrix or Vector{θvec} of posterior draws (each draw is a θ-vector or EntropyPriorParams)
returns:
 - region_mean :: Vector{Float64}
 - region_ci_low, region_ci_high :: Vector{Float64}
"""
function posterior_to_region_map(θ_samples, nodes, region_list; α=0.05)
    S = size(θ_samples, 1)
    m = length(region_list)
    C = zeros(Float64, m, S)
    for s in 1:S
        θsamp = θ_samples[s, :]
        # convert vector -> EntropyPriorParams if needed, or pass as vector to make_parameterized_prior
        # We'll call a helper that accepts θ vector or struct in Main scope:
        θobj = θsamp  # user can adapt if their make_parameterized_prior expects struct
        π = Main.make_parameterized_prior(nodes, θobj)
        # aggregate per region
        node2region_idx = build_region_index(nodes, region_list)[1]
        region_mass = zeros(Float64, m)
        for i in 1:length(π)
            r = node2region_idx[i]
            if r > 0
                region_mass[r] += π[i]
            end
        end
        if sum(region_mass) > 0
            C[:, s] .= region_mass ./ sum(region_mass)
        else
            C[:, s] .= 1.0 / m
        end
    end
    region_mean = mean(C, dims=2)[:]
    region_ci_low = mapslices(x->quantile(x, α/2), C; dims=2)[:]
    region_ci_high = mapslices(x->quantile(x, 1-α/2), C; dims=2)[:]
    return region_mean, region_ci_low, region_ci_high
end


# ---------------------------
# Training wrappers
# ---------------------------
# Preferred: use Python's 'sbi' via PyCall for proper NPE (if available).
function train_posterior_py_sbi(X::AbstractMatrix, Y::AbstractMatrix; pyenv_python::Union{Nothing,String}=nothing,
                                savepath::String="posterior_sbi.pt", epochs=50, batch=128)
    try
        using PyCall
    catch
        error("PyCall is required for train_posterior_py_sbi. Add PyCall.jl and configure Python environment.")
    end
    # Optionally set PYTHON path (user can set ENV["PYTHON"] before starting Julia)
    py = pyimport("numpy")
    torch = pyimport("torch")
    # Minimal pipeline: convert X,Y -> numpy and call sbi inference (user must have sbi installed)
    np = pyimport("numpy")
    try
        sbi = pyimport("sbi")
    catch
        error("Python package 'sbi' not found. Install it (pip install sbi) in the Python env used by PyCall.")
    end

    X_np = np.array(X; dtype=np.float32)
    Y_np = np.array(Y; dtype=np.float32)

    # Python-side script: we will create a simple sbi inference wrapper using SNPE_C
    py"""
import numpy as np
import torch
from sbi import utils as sbi_utils
from sbi.inference import SNPE_C
"""
    prior = nothing  # SNPE can be used with a broad prior; user can supply a torch prior if desired
    # Wrap data into torch tensors / dataset
    # For brevity we hand off to user to expand; return a placeholder file name
    println("[train_posterior_py_sbi] Converted data and returned control to Python; implement training in Python side for full NPE.")
    return savepath
end


# Fallback: Flux ensemble regressor (returns ensemble predictions as approximate posterior)
function train_posterior_flux_ensemble(X::Matrix{Float32}, Y::Matrix{Float32};
                                       nmodels::Int=5, nepochs::Int=100, batchsize::Int=256,
                                       lr::Float64=1e-3, val_frac::Float64=0.1, rng=Random.GLOBAL_RNG)
    N, D = size(X)
    θdim = size(Y,2)
    # simple standardization
    μx = mean(X, dims=1)
    σx = std(X, dims=1) .+ 1e-6
    Xs = (X .- repeat(μx, N, 1)) ./ repeat(σx, N, 1)

    # train/val split
    idx = collect(1:N)
    shuffle!(rng, idx)
    nval = Int(round(val_frac * N))
    validx = idx[1:nval]; trainidx = idx[(nval+1):end]

    models = Vector{Any}(undef, nmodels)
    optims = Vector{Any}(undef, nmodels)

    for m in 1:nmodels
        model = Chain(
            Dense(D, 128, relu),
            Dense(128, 128, relu),
            Dense(128, θdim)
        )
        opt = ADAM(lr)
        batches = MLDataPattern.eachbatch((Xs[trainidx, :]', Y[trainidx, :]'), batchsize)
        for epoch in 1:nepochs
            for (xb, yb) in batches
                xb = xb |> gpu_or_cpu(model)
                yb = yb |> gpu_or_cpu(model)
                gs = gradient(params(model)) do
                    ŷ = model(xb)
                    loss = Flux.Losses.mse(ŷ, yb)
                    return loss
                end
                Flux.Optimise.update!(opt, params(model), gs)
            end
        end
        models[m] = model
        optims[m] = opt
    end

    # Return ensemble + preprocessing params for inference
    return Dict(:models => models, :μx => μx, :σx => σx, :θdim => θdim)
end

# small helper to place data on cpu/gpu depending if model has gpu layers
gpu_or_cpu(x) = x  # placeholder; avoid GPU specifics in this snippet

end # module VBI
