# ═══════════════════════════════════════════════════════════════════════════════
#                              MIRAGE - Neural Network Probing
# ═══════════════════════════════════════════════════════════════════════════════
# Tools for analyzing internal neural network representations
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
#                              NEURON ANALYSIS
# ───────────────────────────────────────────────────────────────────────────────

"""
    probe_neurons(model, x::Array, layer_name::String)

Get neuron activations for a specific layer.
"""
function probe_neurons(model, x::Array, layer_name::String)::LayerProbe
    log_info("Probing layer: $layer_name")
    
    # Get layer output
    activations = get_layer_output(model, x, layer_name)
    
    # Compute statistics
    stats = compute_activation_stats(activations)
    
    # Find most activated neurons
    flat_activations = vec(activations)
    sorted_indices = sortperm(flat_activations, rev=true)
    top_neurons = [(i, flat_activations[i]) for i in sorted_indices[1:min(10, length(sorted_indices))]]
    
    return LayerProbe(
        layer_name,
        activations,
        stats,
        top_neurons
    )
end

"""
    most_activated(model, x::Array; top_k::Int = 10, layers::Union{Vector{String}, Nothing} = nothing)

Find most activated neurons across layers.
"""
function most_activated(model, x::Array;
                        top_k::Int = 10,
                        layers::Union{Vector{String}, Nothing} = nothing)::Vector{NeuronActivation}
    
    log_info("Finding top $top_k activated neurons...")
    
    # Get all layers if not specified
    if layers === nothing
        layers = get_layer_names(model)
    end
    
    all_neurons = NeuronActivation[]
    
    for layer in layers
        try
            activations = get_layer_output(model, x, layer)
            
            # Flatten and track indices
            flat = vec(activations)
            for (i, val) in enumerate(flat)
                coord = ind2sub_custom(size(activations), i)
                push!(all_neurons, NeuronActivation(layer, coord, val))
            end
        catch e
            log_warning("Could not probe layer $layer: $e")
        end
    end
    
    # Sort by activation value
    sort!(all_neurons, by=n->n.value, rev=true)
    
    return all_neurons[1:min(top_k, length(all_neurons))]
end

"""Convert linear index to coordinates."""
function ind2sub_custom(dims::Tuple, idx::Int)::Tuple
    coords = []
    for d in reverse(dims)
        push!(coords, ((idx - 1) % d) + 1)
        idx = (idx - 1) ÷ d + 1
    end
    return Tuple(reverse(coords))
end

# ───────────────────────────────────────────────────────────────────────────────
#                              ACTIVATION MAXIMIZATION
# ───────────────────────────────────────────────────────────────────────────────

"""
    maximize_activation(model, layer_name::String, neuron_idx;
                        iterations::Int = 100, lr::Float64 = 0.1,
                        regularization::Float64 = 0.01)

Find input that maximizes specific neuron activation.
"""
function maximize_activation(model, layer_name::String, neuron_idx;
                             iterations::Int = 100,
                             lr::Float64 = 0.1,
                             regularization::Float64 = 0.01)::Array{Float32}
    
    log_info("Maximizing activation of neuron $neuron_idx in $layer_name...")
    
    input_shape = model.input_shape
    
    # Start with random noise
    x = randn(Float32, input_shape...) .* 0.1f0
    
    for iter in 1:iterations
        # Get current activation
        activations = get_layer_output(model, x, layer_name)
        current_val = activations[neuron_idx...]
        
        # Compute gradient w.r.t input (numerical)
        h = 1e-4f0
        grad = zeros(Float32, size(x))
        
        for i in eachindex(x)
            x_plus = copy(x)
            x_minus = copy(x)
            x_plus[i] += h
            x_minus[i] -= h
            
            act_plus = get_layer_output(model, x_plus, layer_name)[neuron_idx...]
            act_minus = get_layer_output(model, x_minus, layer_name)[neuron_idx...]
            
            grad[i] = (act_plus - act_minus) / (2h)
        end
        
        # Update input (gradient ascent)
        x .+= lr .* grad
        
        # Regularization (L2)
        x .-= regularization .* x
        
        # Clamp to valid range
        x = clamp.(x, -1.0f0, 1.0f0)
        
        if iter % 20 == 0
            log_info("  Iteration $iter: activation = $(round(current_val, digits=4))")
        end
    end
    
    log_success("Activation maximization complete")
    
    return x
end

"""
    feature_visualization(model, layer_name::String; 
                          n_features::Int = 9, iterations::Int = 100)

Generate visualizations for multiple features in a layer.
"""
function feature_visualization(model, layer_name::String;
                               n_features::Int = 9,
                               iterations::Int = 100)::Vector{Array{Float32}}
    
    log_info("Generating $n_features feature visualizations for $layer_name...")
    
    visualizations = Vector{Array{Float32}}()
    
    # Get layer dimensions
    test_input = zeros(Float32, model.input_shape...)
    layer_output = get_layer_output(model, test_input, layer_name)
    n_total = length(layer_output)
    
    # Select features to visualize
    feature_indices = round.(Int, range(1, n_total, length=n_features))
    
    for (i, idx) in enumerate(feature_indices)
        coord = ind2sub_custom(size(layer_output), idx)
        vis = maximize_activation(model, layer_name, coord, iterations=iterations)
        push!(visualizations, vis)
        log_info("  Feature $i/$n_features complete")
    end
    
    return visualizations
end

# ───────────────────────────────────────────────────────────────────────────────
#                              REPRESENTATION ANALYSIS
# ───────────────────────────────────────────────────────────────────────────────

"""
    representation_similarity(model, x1::Array, x2::Array, layer_name::String)

Compute representation similarity between two inputs.
"""
function representation_similarity(model, x1::Array, x2::Array, 
                                   layer_name::String)::Float64
    
    # Get representations
    rep1 = vec(get_layer_output(model, x1, layer_name))
    rep2 = vec(get_layer_output(model, x2, layer_name))
    
    # Cosine similarity
    similarity = dot(rep1, rep2) / (norm(rep1) * norm(rep2) + 1e-10)
    
    return similarity
end

"""
    centered_kernel_alignment(model, X::Vector, layer_name::String)

Compute CKA (Kornblith et al., 2019) for representations.
"""
function centered_kernel_alignment(model, X::Vector, layer_name::String)::Matrix{Float64}
    n = length(X)
    
    # Get representations
    representations = [vec(get_layer_output(model, x, layer_name)) for x in X]
    
    # Compute kernel matrices
    K = zeros(n, n)
    for i in 1:n
        for j in 1:n
            K[i, j] = dot(representations[i], representations[j])
        end
    end
    
    # Center kernel
    K_centered = center_kernel(K)
    
    return K_centered
end

"""Center a kernel matrix."""
function center_kernel(K::Matrix)::Matrix{Float64}
    n = size(K, 1)
    H = I - ones(n, n) / n
    return H * K * H
end

"""
    layer_similarity_matrix(model, x::Array)

Compute pairwise similarity between all layers.
"""
function layer_similarity_matrix(model, x::Array)::Dict{String, Any}
    layers = get_layer_names(model)
    n_layers = length(layers)
    
    # Get all representations
    representations = Dict{String, Vector{Float64}}()
    for layer in layers
        try
            representations[layer] = vec(get_layer_output(model, x, layer))
        catch
            continue
        end
    end
    
    valid_layers = collect(keys(representations))
    n = length(valid_layers)
    
    # Compute similarity matrix
    similarity = zeros(n, n)
    for (i, l1) in enumerate(valid_layers)
        for (j, l2) in enumerate(valid_layers)
            r1, r2 = representations[l1], representations[l2]
            similarity[i, j] = dot(r1, r2) / (norm(r1) * norm(r2) + 1e-10)
        end
    end
    
    return Dict(
        "layers" => valid_layers,
        "similarity" => similarity
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              CONCEPT PROBING
# ───────────────────────────────────────────────────────────────────────────────

"""
    probe_concept(model, concept_samples::Vector, non_concept_samples::Vector,
                  layer_name::String)

Train linear probe for a concept.
"""
function probe_concept(model, concept_samples::Vector, non_concept_samples::Vector,
                       layer_name::String)::Dict{String, Any}
    
    log_info("Training concept probe on $layer_name...")
    
    # Get representations
    pos_reps = [vec(get_layer_output(model, x, layer_name)) for x in concept_samples]
    neg_reps = [vec(get_layer_output(model, x, layer_name)) for x in non_concept_samples]
    
    # Create training data
    X = vcat(pos_reps, neg_reps)
    X_matrix = reduce(hcat, X)'
    y = vcat(ones(length(pos_reps)), zeros(length(neg_reps)))
    
    # Train logistic regression (simplified)
    n_features = size(X_matrix, 2)
    weights = zeros(n_features)
    bias = 0.0
    lr = 0.01
    
    for epoch in 1:100
        for i in 1:length(y)
            pred = sigmoid(dot(X_matrix[i, :], weights) + bias)
            error = y[i] - pred
            weights .+= lr .* error .* X_matrix[i, :]
            bias += lr * error
        end
    end
    
    # Evaluate
    predictions = [sigmoid(dot(X_matrix[i, :], weights) + bias) > 0.5 for i in 1:length(y)]
    accuracy = mean(predictions .== y)
    
    log_success("Concept probe accuracy: $(round(accuracy * 100, digits=1))%")
    
    return Dict(
        "weights" => weights,
        "bias" => bias,
        "accuracy" => accuracy,
        "layer" => layer_name
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              ACTIVATION STATISTICS
# ───────────────────────────────────────────────────────────────────────────────

"""Compute activation statistics."""
function compute_activation_stats(activations::Array)::Dict{String, Float64}
    flat = vec(activations)
    
    return Dict(
        "mean" => mean(flat),
        "std" => std(flat),
        "min" => minimum(flat),
        "max" => maximum(flat),
        "sparsity" => mean(flat .== 0),
        "l1_norm" => sum(abs.(flat)),
        "l2_norm" => sqrt(sum(flat.^2))
    )
end

"""
    layer_statistics(model, X::Vector)

Compute statistics across multiple samples.
"""
function layer_statistics(model, X::Vector)::Dict{String, Dict}
    layers = get_layer_names(model)
    stats = Dict{String, Dict}()
    
    for layer in layers
        try
            all_stats = [compute_activation_stats(get_layer_output(model, x, layer)) for x in X]
            
            # Aggregate
            stats[layer] = Dict(
                "mean" => mean([s["mean"] for s in all_stats]),
                "std" => mean([s["std"] for s in all_stats]),
                "sparsity" => mean([s["sparsity"] for s in all_stats])
            )
        catch
            continue
        end
    end
    
    return stats
end

# ───────────────────────────────────────────────────────────────────────────────
#                              DISPLAY
# ───────────────────────────────────────────────────────────────────────────────

"""
    display_probe(probe::LayerProbe)

Display layer probe results.
"""
function display_probe(probe::LayerProbe)
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                      LAYER PROBE                             ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ Layer:        $(rpad(probe.layer_name, 45))║")
    println("║ Shape:        $(rpad(string(size(probe.activations)), 45))║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ Statistics:                                                  ║")
    println("║   Mean:       $(rpad(round(probe.statistics["mean"], digits=4), 45))║")
    println("║   Std:        $(rpad(round(probe.statistics["std"], digits=4), 45))║")
    println("║   Min:        $(rpad(round(probe.statistics["min"], digits=4), 45))║")
    println("║   Max:        $(rpad(round(probe.statistics["max"], digits=4), 45))║")
    println("║   Sparsity:   $(rpad(round(probe.statistics["sparsity"] * 100, digits=1), 42))% ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ Top Activated Neurons:                                       ║")
    
    for (i, (idx, val)) in enumerate(probe.top_neurons[1:min(5, length(probe.top_neurons))])
        println("║   $i. Index $(rpad(idx, 10)) Value: $(rpad(round(val, digits=4), 20))║")
    end
    
    println("╚══════════════════════════════════════════════════════════════╝")
    println()
end

"""
    display_neurons(neurons::Vector{NeuronActivation}; top_k::Int = 10)

Display most activated neurons.
"""
function display_neurons(neurons::Vector{NeuronActivation}; top_k::Int = 10)
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                  MOST ACTIVATED NEURONS                      ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    
    for (i, neuron) in enumerate(neurons[1:min(top_k, length(neurons))])
        layer_str = rpad(neuron.layer, 20)
        coord_str = rpad(string(neuron.coordinate), 15)
        val_str = rpad(round(neuron.value, digits=4), 12)
        println("║ $i. $(layer_str) $(coord_str) $(val_str)║")
    end
    
    println("╚══════════════════════════════════════════════════════════════╝")
    println()
end
