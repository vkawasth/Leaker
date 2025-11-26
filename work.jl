using BSON, VBI

# 1) load sims produced earlier
BSON.@load "sbi_dataset.bson" θs sims

# 2) define region_list used in your project (must match what make_region_prior uses)
region_list = [
    "ACA","AI","AOB","AOBgr","AON","AUD","BLA","BMA","BS","CA1sp","CB","CBXmo",
    "CNU","COA","CTXsp","CUL4","DORpm","DORsm","DP","ECT","EP","FN","FRP","GU",
    "HB","HPF","HY","ILA","LA","LSX","LZ","MB","MBmot","MBsen","MEZ","MO","MY",
    "MY-mot","MY-sat","MY-sen","OLF","ORB","P-mot","P-sat","P-sen","PA","PAA",
    "PAL","PALc","PALm","PALv","PAR","PERI","PIR","PL","POST","PRE","PVR","PVZ",
    "RHP","RSP","SNc","SS","STRv","SUB","TEa","TR","TT","VIS","VISC","VS","bgr",
    "fiber tracts","root","sAMY"
]

# 3) build feature matrix
X, Y, meta = VBI.build_features_from_sims(θs, sims; nodes=nodes, region_list=region_list)

# 4) train posterior (fallback)
ensemble = VBI.train_posterior_flux_ensemble(X, Y; nmodels=5, nepochs=40)

# 5) inference on observed x_obs
# x_obs must be built the same way as rows of X (region entropies + other features)
x_obs = X[1,:]  # e.g. for testing; normally compute from observed p_obs
# standardize using μx,σx from ensemble returned dict
μx = ensemble[:μx]; σx = ensemble[:σx]
xstd = (x_obs .- μx[:]) ./ σx[:]

# sample pseudo-posterior via ensemble
S = 200
θ_samples = zeros(Float32, S, ensemble[:θdim])
for s in 1:S
    m = rand(1:length(ensemble[:models]))
    ypred = ensemble[:models][m](xstd')
    θ_samples[s,:] .= ypred |> Array
end

# 6) map posterior samples -> region responsibilities
region_mean, region_ci_low, region_ci_high = VBI.posterior_to_region_map(θ_samples, nodes, region_list)
# rank
ranked = sortperm(region_mean, rev=true)
println("Top regions: ", region_list[ranked[1:10]])
