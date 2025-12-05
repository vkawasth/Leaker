
# ========================================================
# JOYAL’S ARITHMETIC UNIVERSE AS LIST-ARITHMETIC PRETOPOS
# ========================================================
# I used simple julia lists earlier now we implement 3 key ideas of 
# arithmetic universes - Morphisms Node.weight, add_fn and mul_fn 
# these functions can implement non commutative algebra rules
# for each class of nodes (see struct Node
# 

using Base.Threads, Random, Statistics

# ===========================
# 1. Node Definition
# ===========================
mutable struct Node
    id::Int
    state::Float32
    weight::Float32
    add_fn::Function
    mul_fn::Function
end

# ===========================
# 2. Example Node Functions
# ===========================
# Addition functions
standard_add(a, b) = a + b
accentuate_add(a, b) = a + 2.0f0 * b
dampen_add(a, b) = a + 0.5f0 * b

# Multiplication functions
linear_mul(a, s) = a * s
anti_comm_mul(a, b) = -b * a                 # non-commutative multiplication
modular_mul(a, b) = Float32(mod(a * b, 7))  # discrete modulo multiplication

# ===========================
# 3. Initialize Nodes
# ===========================
function init_nodes(n_nodes::Int)
    nodes = Vector{Node}(undef, n_nodes)
    rng = MersenneTwister(42)
    
    add_options = [standard_add, accentuate_add, dampen_add]
    mul_options = [linear_mul, anti_comm_mul, modular_mul]
    
    for i in 1:n_nodes
        weight = rand(rng, Float32)
        add_fn = add_options[rand(rng, 1:3)]
        mul_fn = mul_options[rand(rng, 1:3)]
        nodes[i] = Node(i, 0.0f0, weight, add_fn, mul_fn)
    end
    return nodes
end

# ===========================
# 4. Node Response
# ===========================
function respond!(node::Node, stimulus::Vector{Float32})
    raw_response = sum(node.weight .* stimulus)
    
    # Apply custom addition
    updated_state = node.add_fn(node.state, raw_response)
    
    # Apply custom multiplication (scalar factor = 1.0 for simplicity)
    response = node.mul_fn(updated_state, 1.0f0)
    
    # Update node state
    node.state = response
    
    return response
end

# ===========================
# 5. Categorize Nodes
# ===========================
function categorize_nodes(nodes::Vector{Node}, stimulus::Vector{Float32}, threshold::Float32)
    positives, negatives, neutrals = Int[], Int[], Int[]
    
    @threads for i in 1:length(nodes)
        r = respond(nodes[i], stimulus)
        if r > threshold
            push!(positives, nodes[i].id)
        elseif r < -threshold
            push!(negatives, nodes[i].id)
        else
            push!(neutrals, nodes[i].id)
        end
    end
    
    return positives, negatives, neutrals
end

# =========================================
# 1. Function to get nodes within response ranges
# =========================================
"""
get_response_subsets(nodes, stimulus; top_pos=10, top_neg=20)

Given a stimulus, returns three subsets of node IDs:
1. Positively responding nodes (top_pos highest responses)
2. Negatively responding nodes (top_neg lowest responses)
3. Neutral / remaining nodes (optional)
"""
function get_response_subsets(nodes::Vector{Node}, stimulus::Vector{Float32}; top_pos=10, top_neg=20)
    n_nodes = length(nodes)
    responses = zeros(Float32, n_nodes)
    
    # Compute response for each node (mutates node.state)
    @threads for i in 1:n_nodes
        responses[i] = respond!(nodes[i], stimulus)
    end
    
    # Get indices sorted by response descending
    sorted_indices = sortperm(responses, rev=true)  # largest first
    
    # Positive subset (top N)
    positive_ids = sorted_indices[1:top_pos]
    
    # Negative subset (bottom N)
    negative_ids = sorted_indices[end-top_neg+1:end]
    
    # Neutral / remaining nodes
    all_ids = collect(1:n_nodes)
    neutral_ids = setdiff(all_ids, union(positive_ids, negative_ids))
    
    return positive_ids, negative_ids, neutral_ids
end


# ===========================
# 6. Homogenize Nodes
# ===========================
function homogenize(nodes::Vector{Node}, category_ids::Vector{Int})
    if isempty(category_ids)
        return
    end
    avg_state = mean(nodes[id].state for id in category_ids)
    for id in category_ids
        nodes[id].state = avg_state
    end
end

# Not calling only for debugging
function get_top_nodes(nodes::Vector{Node}, stimulus::Vector{Float32}; top_pos=10, top_neg=20)
    n_nodes = length(nodes)
    responses = zeros(Float32, n_nodes)
    
    @threads for i in 1:n_nodes
        responses[i] = respond!(nodes[i], stimulus)
    end
    
    sorted_indices = sortperm(responses, rev=true)  # descending
    
    positive_ids = sorted_indices[1:min(top_pos, n_nodes)]
    negative_ids = sorted_indices[end-min(top_neg, n_nodes)+1:end]
    neutral_ids = setdiff(1:n_nodes, union(positive_ids, negative_ids))
    
    return positive_ids, negative_ids, neutral_ids, responses
end




function summarize_step(nodes::Vector{Node}, positives::Vector{Int}, negatives::Vector{Int}, neutrals::Vector{Int})
    summary = Dict{String, Any}()
    
    # Counts
    summary["positive_count"] = length(positives)
    summary["negative_count"] = length(negatives)
    summary["neutral_count"]  = length(neutrals)
    
    # Mean state per category
    summary["positive_mean"] = isempty(positives) ? 0.0f0 : mean(nodes[id].state for id in positives)
    summary["negative_mean"] = isempty(negatives) ? 0.0f0 : mean(nodes[id].state for id in negatives)
    summary["neutral_mean"]  = isempty(neutrals)  ? 0.0f0 : mean(nodes[id].state for id in neutrals)
    
    # Optional: min/max per category
    summary["positive_min"] = isempty(positives) ? 0.0f0 : minimum(nodes[id].state for id in positives)
    summary["positive_max"] = isempty(positives) ? 0.0f0 : maximum(nodes[id].state for id in positives)
    
    summary["negative_min"] = isempty(negatives) ? 0.0f0 : minimum(nodes[id].state for id in negatives)
    summary["negative_max"] = isempty(negatives) ? 0.0f0 : maximum(nodes[id].state for id in negatives)
    
    summary["neutral_min"] = isempty(neutrals) ? 0.0f0 : minimum(nodes[id].state for id in neutrals)
    summary["neutral_max"] = isempty(neutrals) ? 0.0f0 : maximum(nodes[id].state for id in neutrals)
    
    return summary
end

# ===========================
# 7. Multi-Stimulus Simulation
# ===========================

function simulate_nodes_with_summary!(nodes::Vector{Node}, stimuli::Vector{Vector{Float32}}, threshold::Float32; top_pos=10, top_neg=20, homogenize_each_step=false)
    n_nodes = length(nodes)
    n_steps = length(stimuli)
    
    # History arrays
    category_history = Array{Int8}(undef, n_nodes, n_steps)
    summaries = Vector{Dict{String, Any}}(undef, n_steps)
    top_nodes_history = Vector{Dict{String, Vector{Int}}}(undef, n_steps)
    
    for t in 1:n_steps
        stimulus = stimuli[t]
        
        # --- Step 1: Compute all responses once ---
        responses = zeros(Float32, n_nodes)
        @threads for i in 1:n_nodes
            responses[i] = respond!(nodes[i], stimulus)
        end
        
        # --- Step 2: Determine top positive and negative nodes ---
        sorted_indices = sortperm(responses, rev=true)
        positive_ids = sorted_indices[1:min(top_pos, n_nodes)]
        negative_ids = sorted_indices[end-min(top_neg, n_nodes)+1:end]
        neutral_ids = setdiff(1:n_nodes, union(positive_ids, negative_ids))
        
        # --- Step 3: Categorize all nodes ---
        # Thread-safe assignment
        for i in 1:n_nodes
            if i in positive_ids
                category_history[i, t] = 1
            elseif i in negative_ids
                category_history[i, t] = -1
            else
                category_history[i, t] = 0
            end
        end
        
        # --- Step 4: Optional homogenization ---
        if homogenize_each_step
            homogenize(nodes, positive_ids)
            homogenize(nodes, negative_ids)
            homogenize(nodes, neutral_ids)
        end
        
        # --- Step 5: Save top nodes ---
        top_nodes_history[t] = Dict(
            "positive" => positive_ids,
            "negative" => negative_ids,
            "neutral" => neutral_ids
        )
        
        # --- Step 6: Collect summary ---
        summaries[t] = summarize_step(nodes, positive_ids, negative_ids, neutral_ids)
    end
    
    return category_history, summaries, top_nodes_history
end
# Calls get_top_nodes respond, comment out respond on line 268
# only for debugging related to top nodes.
function simulate_nodes_with_summary!(nodes::Vector{Node}, stimuli::Vector{Vector{Float32}}, threshold::Float32; top_pos=10, top_neg=20, homogenize_each_step=false)
    n_nodes = length(nodes)
    n_steps = length(stimuli)
    category_history = Array{Int8}(undef, n_nodes, length(stimuli))
    summaries = Vector{Dict{String, Any}}(undef, length(stimuli))
    top_nodes_history = Vector{Dict{String, Vector{Int}}}(undef, n_steps)
    
    for t in 1:length(stimuli)
        stimulus = stimuli[t]
        positive_ids, negative_ids, neutral_ids, _ = get_top_nodes(nodes, stimulus; top_pos=top_pos, top_neg=top_neg)     
        positives, negatives, neutrals = Int[], Int[], Int[]
        
        @threads for i in 1:n_nodes
            # Optional: node adapts its arithmetic
            # adapt_node!(nodes[i], t)
            # comment out respond if not calling get_top_nodes. 
            # get_top_nodes calls respond.
            #r = respond!(nodes[i], stimulus)
            # above comment makes r undefined so remove all as set.            
            #if r > threshold
            #    push!(positives, nodes[i].id)
            #elseif r < -threshold
            #    push!(negatives, nodes[i].id)
            #else
            #    push!(neutrals, nodes[i].id)
            #end
        end
        
        # Record category for this step
        for id in positives
            category_history[id, t] = 1
        end
        for id in negatives
            category_history[id, t] = -1
        end
        for id in neutrals
            category_history[id, t] = 0
        end
        
        if homogenize_each_step
            homogenize(nodes, positives)
            homogenize(nodes, negatives)
            homogenize(nodes, neutrals)
        end

        # Save top nodes for this step
        top_nodes_history[t] = Dict(
            "positive" => positive_ids,
            "negative" => negative_ids,
            "neutral" => neutral_ids
        )

        # Collect summary for this stimulus
        summaries[t] = summarize_step(nodes, positives, negatives, neutrals)
    end
    
    return category_history, summaries, top_nodes_history
end
## non multi stimulus simulation for debugging only
function simulate_nodes(nodes::Vector{Node}, stimuli::Vector{Vector{Float32}}, threshold::Float32; homogenize_each_step=false)
    n_nodes = length(nodes)
    # Initialize category history: row=node, column=time step
    category_history = Array{Int8}(undef, n_nodes, length(stimuli)) # 1=pos, -1=neg, 0=neutral
    summaries = Vector{Dict{String, Any}}(undef, length(stimuli))
    
    for t in 1:length(stimuli)
        stimulus = stimuli[t]
        positives, negatives, neutrals = categorize_nodes(nodes, stimulus, threshold)
        
        # Record categories for this step
        for id in positives
            category_history[id, t] = 1
        end
        for id in negatives
            category_history[id, t] = -1
        end
        for id in neutrals
            category_history[id, t] = 0
        end
        
        # Optional: homogenize per category
        if homogenize_each_step
            homogenize(nodes, positives)
            homogenize(nodes, negatives)
            homogenize(nodes, neutrals)
        end
    end
    return category_history, summaries
end

# ===========================
# 8. Example Usage
# ===========================
n_nodes = 3_500_000
nodes = init_nodes(n_nodes)

# Generate a sequence of 5 random stimuli
stimuli = [rand(Float32, 10) for _ in 1:5]
threshold = 1.0f0

# Run simulation
# We get top nodes entering leaving stimulus.
category_history, summaries, top_nodes_history = simulate_nodes_with_summary!(nodes, stimuli, threshold, top_pos=10, top_neg=20, homogenize_each_step=true)

# Get top responding nodes
#top_pos_nodes, top_neg_nodes, neutral_nodes = get_response_subsets(nodes, stimuli; top_pos=10, top_neg=20)

#println("Top positive nodes (loudest responses): ", top_pos_nodes)
#println("Top negative nodes (strongest negative responses): ", top_neg_nodes)
#println("Number of neutral / remaining nodes: ", length(neutral_nodes))

println("Top positive nodes at step 1: ", top_nodes_history[1]["positive"])
println("Top negative nodes at step 1: ", top_nodes_history[1]["negative"])


# Quick stats for final step
final_step = size(category_history, 2)
println("Final step category counts:")
println("Positive: ", count(category_history[:, final_step] .== 1))
println("Negative: ", count(category_history[:, final_step] .== -1))
println("Neutral: ", count(category_history[:, final_step] .== 0))

# Print summary for each stimulus
for t in 1:length(stimuli)
    println("Stimulus $t summary: ", summaries[t])
end
