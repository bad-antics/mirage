# ═══════════════════════════════════════════════════════════════════════════════
#                              MIRAGE - Saliency Analysis
# ═══════════════════════════════════════════════════════════════════════════════
# Attribution and saliency methods for understanding model decisions
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
#                              GRADIENT SALIENCY
# ───────────────────────────────────────────────────────────────────────────────

"""
    gradient_saliency(model, x::Array; target_class::Union{Int, Nothing} = nothing,
                      absolute::Bool = true)

Compute vanilla gradient saliency map.
"""
function gradient_saliency(model, x::Array;
                           target_class::Union{Int, Nothing} = nothing,
                           absolute::Bool = true)::SaliencyMap
    
    # Get prediction if no target specified
    if target_class === nothing
        pred = predict(model, x)
        target_class = pred.label
    end
    
    # Compute gradient
    grad_info = get_gradient(model, x, target_class)
    saliency = grad_info.gradient
    
    if absolute
        saliency = abs.(saliency)
    end
    
    # Normalize to [0, 1]
    saliency = normalize_saliency(saliency)
    
    return SaliencyMap(
        saliency,
        :gradient,
        target_class,
        maximum(saliency)
    )
end

"""
    gradient_x_input(model, x::Array; target_class::Union{Int, Nothing} = nothing)

Gradient × Input attribution (Shrikumar et al., 2016).
"""
function gradient_x_input(model, x::Array;
                          target_class::Union{Int, Nothing} = nothing)::SaliencyMap
    
    if target_class === nothing
        pred = predict(model, x)
        target_class = pred.label
    end
    
    grad_info = get_gradient(model, x, target_class)
    saliency = grad_info.gradient .* x
    
    saliency = normalize_saliency(abs.(saliency))
    
    return SaliencyMap(saliency, :gradient_x_input, target_class, maximum(saliency))
end

# ───────────────────────────────────────────────────────────────────────────────
#                              INTEGRATED GRADIENTS
# ───────────────────────────────────────────────────────────────────────────────

"""
    integrated_gradients(model, x::Array; baseline::Union{Array, Nothing} = nothing,
                         target_class::Union{Int, Nothing} = nothing,
                         steps::Int = 50)

Integrated Gradients (Sundararajan et al., 2017).

Computes path integral of gradients from baseline to input.
"""
function integrated_gradients(model, x::Array;
                              baseline::Union{Array, Nothing} = nothing,
                              target_class::Union{Int, Nothing} = nothing,
                              steps::Int = 50)::SaliencyMap
    
    log_info("Computing Integrated Gradients ($steps steps)...")
    
    # Default baseline is zeros
    if baseline === nothing
        baseline = zeros(eltype(x), size(x))
    end
    
    # Get target class
    if target_class === nothing
        pred = predict(model, x)
        target_class = pred.label
    end
    
    # Compute integral
    scaled_gradients = zeros(Float64, size(x))
    
    for step in 0:steps
        # Interpolate between baseline and input
        alpha = step / steps
        x_interp = baseline .+ alpha .* (x .- baseline)
        
        # Get gradient at interpolated point
        grad_info = get_gradient(model, x_interp, target_class)
        scaled_gradients .+= grad_info.gradient
    end
    
    # Average and multiply by (x - baseline)
    integrated = (scaled_gradients ./ (steps + 1)) .* (x .- baseline)
    
    # Take absolute value and normalize
    saliency = normalize_saliency(abs.(integrated))
    
    log_success("Integrated Gradients computed")
    
    return SaliencyMap(saliency, :integrated_gradients, target_class, maximum(saliency))
end

# ───────────────────────────────────────────────────────────────────────────────
#                              SMOOTHGRAD
# ───────────────────────────────────────────────────────────────────────────────

"""
    smoothgrad(model, x::Array; target_class::Union{Int, Nothing} = nothing,
               n_samples::Int = 50, noise_level::Float64 = 0.1)

SmoothGrad (Smilkov et al., 2017).

Averages gradients over noisy samples.
"""
function smoothgrad(model, x::Array;
                    target_class::Union{Int, Nothing} = nothing,
                    n_samples::Int = 50,
                    noise_level::Float64 = 0.1)::SaliencyMap
    
    log_info("Computing SmoothGrad ($n_samples samples)...")
    
    if target_class === nothing
        pred = predict(model, x)
        target_class = pred.label
    end
    
    # Standard deviation for noise
    stdev = noise_level * (maximum(x) - minimum(x))
    
    # Accumulate gradients
    total_grad = zeros(Float64, size(x))
    
    for _ in 1:n_samples
        # Add Gaussian noise
        noise = randn(size(x)) .* stdev
        x_noisy = x .+ noise
        
        # Get gradient
        grad_info = get_gradient(model, x_noisy, target_class)
        total_grad .+= grad_info.gradient
    end
    
    # Average
    saliency = total_grad ./ n_samples
    saliency = normalize_saliency(abs.(saliency))
    
    log_success("SmoothGrad computed")
    
    return SaliencyMap(saliency, :smoothgrad, target_class, maximum(saliency))
end

"""
    smoothgrad_squared(model, x::Array; kwargs...)

SmoothGrad with squared gradients (more robust).
"""
function smoothgrad_squared(model, x::Array;
                            target_class::Union{Int, Nothing} = nothing,
                            n_samples::Int = 50,
                            noise_level::Float64 = 0.1)::SaliencyMap
    
    if target_class === nothing
        pred = predict(model, x)
        target_class = pred.label
    end
    
    stdev = noise_level * (maximum(x) - minimum(x))
    total_grad_sq = zeros(Float64, size(x))
    
    for _ in 1:n_samples
        noise = randn(size(x)) .* stdev
        x_noisy = x .+ noise
        
        grad_info = get_gradient(model, x_noisy, target_class)
        total_grad_sq .+= grad_info.gradient.^2
    end
    
    saliency = sqrt.(total_grad_sq ./ n_samples)
    saliency = normalize_saliency(saliency)
    
    return SaliencyMap(saliency, :smoothgrad_squared, target_class, maximum(saliency))
end

# ───────────────────────────────────────────────────────────────────────────────
#                              LIME
# ───────────────────────────────────────────────────────────────────────────────

"""
    lime(model, x::Array; target_class::Union{Int, Nothing} = nothing,
         n_samples::Int = 1000, kernel_width::Float64 = 0.25,
         segments::Int = 50)

LIME - Local Interpretable Model-agnostic Explanations (Ribeiro et al., 2016).
"""
function lime(model, x::Array;
              target_class::Union{Int, Nothing} = nothing,
              n_samples::Int = 1000,
              kernel_width::Float64 = 0.25,
              segments::Int = 50)::SaliencyMap
    
    log_info("Computing LIME explanation ($n_samples samples)...")
    
    if target_class === nothing
        pred = predict(model, x)
        target_class = pred.label
    end
    
    # Create superpixel segmentation
    segment_mask = create_segments(x, segments)
    n_segments = maximum(segment_mask)
    
    # Generate perturbed samples
    samples = zeros(n_samples, n_segments)
    predictions = zeros(n_samples)
    
    for i in 1:n_samples
        # Random binary mask for segments
        mask = rand(Bool, n_segments)
        samples[i, :] = mask
        
        # Create perturbed image
        x_perturbed = perturb_by_segments(x, segment_mask, mask)
        
        # Get prediction
        pred = predict(model, x_perturbed)
        predictions[i] = pred.probabilities[target_class]
    end
    
    # Compute distances for weighting
    original_rep = ones(n_segments)
    distances = [sqrt(sum((samples[i, :] .- original_rep).^2)) for i in 1:n_samples]
    weights = exp.(-distances.^2 ./ kernel_width^2)
    
    # Fit weighted linear model
    coefficients = fit_weighted_linear(samples, predictions, weights)
    
    # Map coefficients back to image
    saliency = zeros(size(x))
    for seg in 1:n_segments
        saliency[segment_mask .== seg] .= coefficients[seg]
    end
    
    saliency = normalize_saliency(saliency)
    
    log_success("LIME explanation computed")
    
    return SaliencyMap(saliency, :lime, target_class, maximum(saliency))
end

# ───────────────────────────────────────────────────────────────────────────────
#                              SHAP (Simplified)
# ───────────────────────────────────────────────────────────────────────────────

"""
    kernel_shap(model, x::Array; target_class::Union{Int, Nothing} = nothing,
                n_samples::Int = 500, segments::Int = 20)

Kernel SHAP (Lundberg & Lee, 2017).

Simplified implementation using LIME-style sampling with SHAP kernel.
"""
function kernel_shap(model, x::Array;
                     target_class::Union{Int, Nothing} = nothing,
                     n_samples::Int = 500,
                     segments::Int = 20)::SaliencyMap
    
    log_info("Computing SHAP values ($n_samples samples)...")
    
    if target_class === nothing
        pred = predict(model, x)
        target_class = pred.label
    end
    
    segment_mask = create_segments(x, segments)
    n_segments = maximum(segment_mask)
    
    # Sample coalitions with SHAP weighting
    samples = zeros(n_samples, n_segments)
    predictions = zeros(n_samples)
    shap_weights = zeros(n_samples)
    
    for i in 1:n_samples
        # Sample subset size
        k = rand(1:n_segments-1)
        
        # Random k features
        mask = zeros(Bool, n_segments)
        selected = shuffle(1:n_segments)[1:k]
        mask[selected] .= true
        
        samples[i, :] = mask
        
        # SHAP kernel weight
        shap_weights[i] = shap_kernel_weight(n_segments, k)
        
        # Perturb and predict
        x_perturbed = perturb_by_segments(x, segment_mask, mask)
        pred = predict(model, x_perturbed)
        predictions[i] = pred.probabilities[target_class]
    end
    
    # Fit weighted linear model
    coefficients = fit_weighted_linear(samples, predictions, shap_weights)
    
    # Map to image
    saliency = zeros(size(x))
    for seg in 1:n_segments
        saliency[segment_mask .== seg] .= coefficients[seg]
    end
    
    saliency = normalize_saliency(abs.(saliency))
    
    log_success("SHAP values computed")
    
    return SaliencyMap(saliency, :shap, target_class, maximum(saliency))
end

"""SHAP kernel weight."""
function shap_kernel_weight(n_features::Int, subset_size::Int)::Float64
    if subset_size == 0 || subset_size == n_features
        return 1e6  # Very high weight for edge cases
    end
    
    return (n_features - 1) / (binomial(n_features, subset_size) * subset_size * (n_features - subset_size))
end

# ───────────────────────────────────────────────────────────────────────────────
#                              GUIDED BACKPROP
# ───────────────────────────────────────────────────────────────────────────────

"""
    guided_backprop(model, x::Array; target_class::Union{Int, Nothing} = nothing)

Guided Backpropagation (Springenberg et al., 2014).

Only backpropagates positive gradients through ReLUs.
"""
function guided_backprop(model, x::Array;
                         target_class::Union{Int, Nothing} = nothing)::SaliencyMap
    
    if target_class === nothing
        pred = predict(model, x)
        target_class = pred.label
    end
    
    # Compute gradient with guided ReLU (simplified simulation)
    grad_info = get_gradient(model, x, target_class)
    guided_grad = grad_info.gradient
    
    # Mask out negative gradients (simulating guided backprop effect)
    guided_grad = max.(guided_grad, 0)
    
    saliency = normalize_saliency(guided_grad)
    
    return SaliencyMap(saliency, :guided_backprop, target_class, maximum(saliency))
end

# ───────────────────────────────────────────────────────────────────────────────
#                              HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

"""Normalize saliency map to [0, 1]."""
function normalize_saliency(saliency::Array)::Array{Float64}
    min_val = minimum(saliency)
    max_val = maximum(saliency)
    
    if max_val - min_val < 1e-10
        return zeros(Float64, size(saliency))
    end
    
    return Float64.((saliency .- min_val) ./ (max_val - min_val))
end

"""Create superpixel segments (simplified grid-based)."""
function create_segments(x::Array, n_segments::Int)::Array{Int}
    dims = size(x)[1:2]
    grid_size = ceil(Int, sqrt(n_segments))
    
    segment_mask = zeros(Int, dims)
    
    seg_h = dims[1] ÷ grid_size
    seg_w = dims[2] ÷ grid_size
    
    seg_id = 0
    for i in 1:grid_size
        for j in 1:grid_size
            seg_id += 1
            
            r_start = (i-1) * seg_h + 1
            r_end = min(i * seg_h, dims[1])
            c_start = (j-1) * seg_w + 1
            c_end = min(j * seg_w, dims[2])
            
            segment_mask[r_start:r_end, c_start:c_end] .= seg_id
        end
    end
    
    return segment_mask
end

"""Perturb image by zeroing out segments."""
function perturb_by_segments(x::Array, segment_mask::Array{Int}, 
                             active_segments::Vector{Bool})::Array
    
    x_perturbed = copy(x)
    n_segments = length(active_segments)
    
    for seg in 1:n_segments
        if !active_segments[seg]
            # Zero out this segment
            mask = segment_mask .== seg
            if ndims(x) == 3
                for c in 1:size(x, 3)
                    x_perturbed[:, :, c][mask] .= 0
                end
            else
                x_perturbed[mask] .= 0
            end
        end
    end
    
    return x_perturbed
end

"""Fit weighted linear regression."""
function fit_weighted_linear(X::Matrix, y::Vector, weights::Vector)::Vector{Float64}
    n, p = size(X)
    
    # Add intercept
    X_aug = hcat(ones(n), X)
    
    # Weighted least squares
    W = Diagonal(weights)
    
    try
        coeffs = (X_aug' * W * X_aug + 1e-6 * I) \ (X_aug' * W * y)
        return coeffs[2:end]  # Skip intercept
    catch
        # Fallback
        return zeros(p)
    end
end

# ───────────────────────────────────────────────────────────────────────────────
#                              COMPARISON
# ───────────────────────────────────────────────────────────────────────────────

"""
    compare_saliency(model, x::Array; methods::Vector{Symbol} = [:gradient, :integrated_gradients, :smoothgrad])

Compare multiple saliency methods.
"""
function compare_saliency(model, x::Array;
                          methods::Vector{Symbol} = [:gradient, :integrated_gradients, :smoothgrad])::Dict{Symbol, SaliencyMap}
    
    results = Dict{Symbol, SaliencyMap}()
    
    for method in methods
        if method == :gradient
            results[method] = gradient_saliency(model, x)
        elseif method == :integrated_gradients
            results[method] = integrated_gradients(model, x)
        elseif method == :smoothgrad
            results[method] = smoothgrad(model, x)
        elseif method == :lime
            results[method] = lime(model, x)
        elseif method == :shap
            results[method] = kernel_shap(model, x)
        elseif method == :guided_backprop
            results[method] = guided_backprop(model, x)
        end
    end
    
    return results
end

"""
    display_saliency(smap::SaliencyMap)

Display saliency map information.
"""
function display_saliency(smap::SaliencyMap)
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                      SALIENCY MAP                            ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ Method:       $(rpad(string(smap.method), 45))║")
    println("║ Target Class: $(rpad(smap.target_class, 45))║")
    println("║ Max Value:    $(rpad(round(smap.max_value, digits=4), 45))║")
    println("║ Shape:        $(rpad(string(size(smap.saliency)), 45))║")
    
    # Show intensity histogram
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ Intensity Distribution:                                      ║")
    
    hist = compute_histogram(smap.saliency, 10)
    for (bin, count) in hist
        bar_width = round(Int, count * 40)
        bar = repeat("█", bar_width)
        println("║ $(rpad(bin, 8)) $(rpad(bar, 42))║")
    end
    
    println("╚══════════════════════════════════════════════════════════════╝")
    println()
end

"""Compute histogram bins."""
function compute_histogram(arr::Array, n_bins::Int)::Vector{Tuple{String, Float64}}
    flat = vec(arr)
    bins = range(0, 1, length=n_bins+1)
    
    result = Tuple{String, Float64}[]
    total = length(flat)
    
    for i in 1:n_bins
        count = sum(bins[i] .<= flat .< bins[i+1])
        bin_label = "$(round(bins[i], digits=1))-$(round(bins[i+1], digits=1))"
        push!(result, (bin_label, count / total))
    end
    
    return result
end
