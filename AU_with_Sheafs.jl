using Random, Statistics, LinearAlgebra

# =================================================================
# 1. PARAMETERS & BV/SHEAF CONSTANTS
# =================================================================
n_nodes = 3_500         # Reduced nodes for faster LinearAlgebra on my local machine
n_basis = 5             # Dimension of the C0 state vector p_i (functional complexity)
n_steps = 5             # Simulation steps
n_stimulus = 10         # Dimensions for random stimulus
stability_threshold = 0.1f0 # BV: Minimum singular value threshold for instability (sigma_min(J) -> 0)
coherence_tau = 0.5f0   # Coherence: d_Coh threshold for functional similarity

rng = MersenneTwister(42)

# =================================================================
# 2. AU STATE INITIALIZATION (C0 State and Coherence Metric)
# =================================================================
# C0 State vector p_i: n_nodes x n_basis
# Interpreted as latent parameters for the local probabilistic outcome (Stalk F_x)
states = rand(rng, Float32, n_nodes, n_basis) * 2f0 .- 1f0
initial_states = copy(states)

# Imputed Fisher Information Matrix (I)
# Represents the local system's sensitivity to changes in the C0 basis.
I_matrix = Diagonal(rand(rng, Float32, n_basis) * 0.5f0 .+ 0.1f0)

# =================================================================
# 3. TEST NEIGHBORHOOD FOR JACOBIAN (Exceptional Locus E)
# =================================================================
# U_test simulates the local Representable Neighborhood (U_i) where collapse occurs.
U_test_size = 50
U_test_indices = 1:U_test_size
J_dim = U_test_size * n_basis # Dimension of the local Jacobian

# =================================================================
# 4. PULLBACK FUNCTOR DEFINITION (Node-Specific Alpha)
# =================================================================
function perform_pullback_element_wise(collapsed_states, initial_states, U_indices, alpha_vec)
    """
    Implements the stable Pullback Functor (pi*) using an element-wise 
    alpha vector (alpha_i) to restore functional complexity.
    """
    restored_states = copy(collapsed_states)
    
    for (idx, i) in enumerate(U_indices)
        # alpha_i is the stabilizer for this specific node
        alpha_i = alpha_vec[idx]
        
        # Calculate the "lost complexity" (The fiber E = initial - collapsed)
        lost_complexity = initial_states[i, :] .- restored_states[i, :]
        
        # Apply the restoration: restored = collapsed + alpha_i * (lost complexity)
        restored_states[i, :] += alpha_i * lost_complexity
    end
    
    return restored_states
end

# =================================================================
# 4. PULLBACK FUNCTOR DEFINITION (Must be defined before use)
# =================================================================
function perform_pullback_fixed_scale3(collapsed_states, initial_states, U_indices; alpha=0.05f0)
    """
    Implements the stable Pullback Functor (pi*) to restore functional complexity.
    Uses bounded scaling (alpha) to satisfy the Crepant Condition.
    """
    restored_states = copy(collapsed_states)
    
    for i in U_indices
        # Calculate the "lost complexity" (The fiber E = initial - collapsed)
        lost_complexity = initial_states[i, :] .- restored_states[i, :]
        
        # Apply the restoration: restored = collapsed + alpha * (lost complexity)
        restored_states[i, :] += alpha * lost_complexity
    end
    
    return restored_states
end


# =================================================================
# 5. MAIN SIMULATION LOOP (Applying Functional Flux & Collapse pi_*)
# =================================================================
println("\n--- SIMULATION (PUSHFORWARD pi_*) ---")
singularity_history = Float32[]
category_hist_thresh = Array{Int8}(undef, n_nodes, n_steps)

for t in 1:n_steps
    # 5.1. Functional Flux (The force driving the system)
    step_stim = sum(rand(rng, Float32, n_stimulus, n_basis) * 2f0 .- 1f0, dims=1)

    # 5.2. Local Dynamics F (The C1 Functor)
    # The non-linear saturation (tanh) prevents numerical blow-up, ensuring stability
    # F(p) = tanh(p * (p + Phi_Func))
    states .= tanh.(states .* (states .+ step_stim))

    # 5.3. BV SINGULARITY DETECTION (Jacobian Check in E)
    # This check identifies when the system is near functional collapse (Blow-Down).
    J_local = zeros(Float32, J_dim, J_dim)
    for i in 1:U_test_size
        deriv_vector = 2.0f0 .* states[U_test_indices[i], :] .+ step_stim[1, :]
        start_idx = (i - 1) * n_basis + 1
        end_idx = i * n_basis
        J_local[start_idx:end_idx, start_idx:end_idx] = Diagonal(deriv_vector)
    end

    sigma_min = minimum(svd(J_local).S)
    push!(singularity_history, sigma_min)

    if sigma_min < stability_threshold
        println("Time $(t): ALGEBRAIC INSTABILITY (Blow-Down) DETECTED! sigma_min(J) = $(round(sigma_min, digits=4)) < $stability_threshold")
    end

    # 5.4. COHERENCE CHECK (Metric collapse)
    i_rand, j_rand = rand(rng, 1:n_nodes, 2)
    p_diff = states[i_rand, :] .- states[j_rand, :]
    d_coh = sqrt(dot(p_diff, I_matrix * p_diff))
    
    println("Step $t: sigma_min=$(round(sigma_min, digits=4)), d_Coh_test=$(round(d_coh, digits=4))")
end
# 'states' is now the collapsed manifold M'

# =================================================================
# 6. RESTORATION CHECK (Pullback pi*) - ADAPTIVE ALPHA LEARNING
# =================================================================
# The adaptive learning attempts proved that:

#    L2 Norm (Euclidean): Too weak.

#    Loss Ratio (Fisher-Rao): Mathematically unstable (α<0).

#    Preserved Ratio (Fisher-Rao): Produces a mathematically sound α≈0.83, but one 
#    that is functionally too weak to achieve high fidelity restoration due to the 
#    non-linear collapse.

#    Key Insight: The optimal α is not the geometrically preserved ratio, but a value 
#    that overcompensates for the observed collapse to counteract numerical and topological 
#    entropy introduced by the tanh dynamics.

#   The failure to achieve Error<0.1 with αmean​≈0.886 is not a failure of the α calculation; 
#   it is a profound limitation of the simple affine Pullback Functor (π∗).

#   The required stabilizer α must be 0.99 to restore the state, but the actual collapse 
#   dynamics (π∗​) only provide the information necessary to calculate an α≈0.886.

#   The Gap (0.99−0.886) represents the information loss that the affine Pullback Functor 
#   cannot recover through simple scaling. This loss is due to the non-linear 
#   transformation F(p)=tanh(p⋅(p+ΦFunc​)). The tanh function destroys information 
#   (topological entropy) that a simple linear functor cannot perfectly invert.

# 6.1. Calculate the Node-Specific Alpha Vector (alpha_vec)

# Define the empirical overcompensation factor
# This factor is crucial for satisfying the Crepant Condition in the non-linear system
OVERCOMPENSATION_BETA = 1.19f0 # Derived from 0.99 / 0.8322

alpha_vec = Float32[]
initial_magnitude_I_vec = Float32[]
collapsed_magnitude_I_vec = Float32[]

# Calculate the Coherence-Aware Magnitude for each node in the Exceptional Locus E
for i in U_test_indices
    
    # 1. Magnitude of the initial state in I-space
    init_mag_I = sqrt(dot(initial_states[i, :], I_matrix * initial_states[i, :]))
    push!(initial_magnitude_I_vec, init_mag_I)
    
    # 2. Magnitude of the collapsed state in I-space (The preserved signal)
    collapsed_mag_I = sqrt(dot(states[i, :], I_matrix * states[i, :]))
    push!(collapsed_magnitude_I_vec, collapsed_mag_I)
    
    # Node-Specific Alpha (Preserved Ratio)
    # This is the base functional requirement for restoration
    alpha_i_ratio = collapsed_mag_I / init_mag_I
    
    # Apply global overcompensation (Beta) and cap at 1.0 (physical constraint)
    alpha_i_learned = min(1.0f0, alpha_i_ratio * OVERCOMPENSATION_BETA)
    
    push!(alpha_vec, alpha_i_learned)
end

final_alpha_mean = mean(alpha_vec)
mean_initial_mag_I = mean(initial_magnitude_I_vec)
mean_collapsed_mag_I = mean(collapsed_magnitude_I_vec)


println("\n--- ADAPTIVE ALPHA LEARNING (ELEMENT-WISE, BETA-CORRECTED) ---")
println("Nodes in E (U_test_size): $(U_test_size)")
println("Mean Collapsed I-Magnitude in E (Preserved): $(round(mean_collapsed_mag_I, digits=6))")
println("Mean Initial I-Magnitude in E (Initial): $(round(mean_initial_mag_I, digits=6))")
println("Correction Factor Beta: $(OVERCOMPENSATION_BETA)")
println("Mean Learned Stabilizer Alpha (Phi_Norcain): $(round(final_alpha_mean, digits=6))")

# 6.2. Final Restoration Check using the Learned Alpha

# Perform Pullback using the final, learned alpha
#restored_states = perform_pullback_fixed_scale3(states, initial_states, U_test_indices; alpha=final_alpha)
# Perform Pullback using the element-wise alpha vector
restored_states = perform_pullback_element_wise(states, initial_states, U_test_indices, alpha_vec)

# 6.3. Check Crepant Condition Fidelity (Coherence Metric)
initial_coherence_sample = Float32[]
restored_coherence = Float32[]

# Sample coherence checks within U_test for fidelity assessment
for k in 1:20 
    i, j = rand(rng, U_test_indices, 2)
    
    # Initial Coherence Check
    p_diff_init = initial_states[i, :] .- initial_states[j, :]
    d_coh_init = sqrt(dot(p_diff_init, I_matrix * p_diff_init))
    push!(initial_coherence_sample, d_coh_init)

    # Restored Coherence Check
    p_diff_restored = restored_states[i, :] .- restored_states[j, :]
    d_coh_restored = sqrt(dot(p_diff_restored, I_matrix * p_diff_restored))
    push!(restored_coherence, d_coh_restored)
end

mean_initial_coherence = mean(initial_coherence_sample)
mean_d_coh_restored = mean(restored_coherence)
fidelity_error = abs(mean_initial_coherence - mean_d_coh_restored)

println("\n--- ALGEBRAIC FIDELITY CHECK (Crepant Condition) ---")
println("Target Mean Initial Coherence (d_Coh) in E: $(round(mean_initial_coherence, digits=6))")
println("Mean Restored Coherence (d_Coh) in E: $(round(mean_d_coh_restored, digits=6))")
println("Fidelity Error (Abs Diff): $(round(fidelity_error, digits=6))")
println("Mean Learned Alpha: $(round(final_alpha_mean, digits=6))")

if fidelity_error < 0.1
    println("\nSUCCESS: Crepant Condition satisfied. Element-Wise Alpha provides topologically faithful restoration.")
else
    # The mean alpha is still printed, but the final decision is based on the vector's performance
    println("\nNOTE: Fidelity error is above 0.1. Even the element-wise alpha with overcompensation is insufficient. Further refinement of the dynamics F(p) or the I-matrix is required.")
end