# ═══════════════════════════════════════════════════════════════════════════════
#                              MIRAGE - Attack Detection
# ═══════════════════════════════════════════════════════════════════════════════
# Methods for detecting adversarial inputs
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
#                              DETECTION METHODS
# ───────────────────────────────────────────────────────────────────────────────

"""
    detect_adversarial(model, x::Array; methods::Vector{Symbol} = [:feature_squeezing, :local_intrinsic_dim])

Detect if input is adversarial using multiple methods.
"""
function detect_adversarial(model, x::Array;
                            methods::Vector{Symbol} = [:feature_squeezing, :local_intrinsic_dim])::Dict{String, Any}
    
    log_info("Running adversarial detection...")
    
    results = Dict{String, Any}()
    scores = Float64[]
    
    for method in methods
        if method == :feature_squeezing
            result = feature_squeezing_detect(model, x)
        elseif method == :local_intrinsic_dim
            result = lid_detect(model, x)
        elseif method == :input_transformation
            result = input_transformation_detect(model, x)
        elseif method == :softmax_entropy
            result = entropy_detect(model, x)
        elseif method == :mahalanobis
            result = mahalanobis_detect(model, x)
        else
            continue
        end
        
        results[string(method)] = result
        push!(scores, result["score"])
    end
    
    # Aggregate scores
    is_adversarial = mean(scores) > 0.5
    confidence = abs(mean(scores) - 0.5) * 2
    
    results["is_adversarial"] = is_adversarial
    results["confidence"] = confidence
    results["aggregate_score"] = mean(scores)
    
    if is_adversarial
        log_warning("Input detected as ADVERSARIAL (confidence: $(round(confidence * 100, digits=1))%)")
    else
        log_success("Input appears CLEAN (confidence: $(round(confidence * 100, digits=1))%)")
    end
    
    return results
end

# ───────────────────────────────────────────────────────────────────────────────
#                              FEATURE SQUEEZING
# ───────────────────────────────────────────────────────────────────────────────

"""
    feature_squeezing_detect(model, x::Array; bit_depth::Int = 4, 
                              filter_size::Int = 3, threshold::Float64 = 0.1)

Feature Squeezing detection (Xu et al., 2017).

Compares predictions before/after squeezing operations.
"""
function feature_squeezing_detect(model, x::Array;
                                   bit_depth::Int = 4,
                                   filter_size::Int = 3,
                                   threshold::Float64 = 0.1)::Dict{String, Any}
    
    # Original prediction
    pred_original = predict(model, x)
    
    # Bit depth reduction
    x_reduced = reduce_bit_depth(x, bit_depth)
    pred_reduced = predict(model, x_reduced)
    
    # Spatial smoothing
    x_smoothed = spatial_smoothing(x, filter_size)
    pred_smoothed = predict(model, x_smoothed)
    
    # Calculate L1 distances
    dist_reduced = sum(abs.(pred_original.probabilities .- pred_reduced.probabilities))
    dist_smoothed = sum(abs.(pred_original.probabilities .- pred_smoothed.probabilities))
    
    max_dist = max(dist_reduced, dist_smoothed)
    
    return Dict(
        "score" => min(max_dist / threshold, 1.0),
        "dist_bit_depth" => dist_reduced,
        "dist_spatial" => dist_smoothed,
        "is_adversarial" => max_dist > threshold,
        "method" => "feature_squeezing"
    )
end

"""Reduce bit depth of image."""
function reduce_bit_depth(x::Array, bits::Int)::Array
    max_val = 2^bits - 1
    return round.(x .* max_val) ./ max_val
end

"""Apply spatial smoothing."""
function spatial_smoothing(x::Array, filter_size::Int)::Array
    # Simple box filter
    if ndims(x) == 3
        result = similar(x)
        pad = filter_size ÷ 2
        
        for c in 1:size(x, 3)
            for i in 1:size(x, 1)
                for j in 1:size(x, 2)
                    i_start = max(1, i - pad)
                    i_end = min(size(x, 1), i + pad)
                    j_start = max(1, j - pad)
                    j_end = min(size(x, 2), j + pad)
                    
                    result[i, j, c] = mean(x[i_start:i_end, j_start:j_end, c])
                end
            end
        end
        return result
    else
        return x  # No smoothing for 1D
    end
end

# ───────────────────────────────────────────────────────────────────────────────
#                              LOCAL INTRINSIC DIMENSIONALITY
# ───────────────────────────────────────────────────────────────────────────────

"""
    lid_detect(model, x::Array; k::Int = 20, n_samples::Int = 100,
               threshold::Float64 = 15.0)

Local Intrinsic Dimensionality detection (Ma et al., 2018).
"""
function lid_detect(model, x::Array;
                    k::Int = 20,
                    n_samples::Int = 100,
                    threshold::Float64 = 15.0)::Dict{String, Any}
    
    # Generate neighbors
    neighbors = [x .+ randn(size(x)) .* 0.01 for _ in 1:n_samples]
    
    # Get representations (last hidden layer)
    layers = get_layer_names(model)
    if isempty(layers)
        return Dict("score" => 0.5, "lid" => 0.0, "method" => "lid")
    end
    
    layer = layers[end]
    
    rep_x = vec(get_layer_output(model, x, layer))
    rep_neighbors = [vec(get_layer_output(model, n, layer)) for n in neighbors]
    
    # Compute distances
    distances = [norm(rep_x .- rep_n) for rep_n in rep_neighbors]
    sort!(distances)
    
    # Take k nearest
    k_distances = distances[1:min(k, length(distances))]
    
    # Compute LID estimate
    lid = estimate_lid(k_distances)
    
    return Dict(
        "score" => min(lid / threshold, 1.0),
        "lid" => lid,
        "is_adversarial" => lid > threshold,
        "method" => "lid"
    )
end

"""Estimate LID using MLE."""
function estimate_lid(distances::Vector{Float64})::Float64
    if isempty(distances) || distances[end] == 0
        return 0.0
    end
    
    r_max = distances[end]
    n = length(distances)
    
    # MLE estimator
    lid = -n / sum(log.(distances ./ r_max .+ 1e-10))
    
    return max(0, lid)
end

# ───────────────────────────────────────────────────────────────────────────────
#                              INPUT TRANSFORMATION
# ───────────────────────────────────────────────────────────────────────────────

"""
    input_transformation_detect(model, x::Array; n_transforms::Int = 10,
                                 threshold::Float64 = 0.3)

Detection based on prediction consistency under transformations.
"""
function input_transformation_detect(model, x::Array;
                                      n_transforms::Int = 10,
                                      threshold::Float64 = 0.3)::Dict{String, Any}
    
    pred_original = predict(model, x)
    original_class = pred_original.label
    
    inconsistent = 0
    
    for _ in 1:n_transforms
        # Apply random transformation
        x_transformed = apply_random_transform(x)
        
        pred = predict(model, x_transformed)
        
        if pred.label != original_class
            inconsistent += 1
        end
    end
    
    inconsistency_rate = inconsistent / n_transforms
    
    return Dict(
        "score" => inconsistency_rate,
        "inconsistency_rate" => inconsistency_rate,
        "is_adversarial" => inconsistency_rate > threshold,
        "method" => "input_transformation"
    )
end

"""Apply random transformation."""
function apply_random_transform(x::Array)::Array
    transform = rand(1:4)
    
    if transform == 1
        # Small rotation (simulated)
        return x .+ randn(size(x)) .* 0.001
    elseif transform == 2
        # Small translation (simulated)
        return circshift(x, (rand(-2:2), rand(-2:2), 0))
    elseif transform == 3
        # Small scaling (simulated)
        scale = 0.95 + 0.1 * rand()
        return x .* scale
    else
        # Add small noise
        return x .+ randn(size(x)) .* 0.005
    end
end

# ───────────────────────────────────────────────────────────────────────────────
#                              ENTROPY-BASED DETECTION
# ───────────────────────────────────────────────────────────────────────────────

"""
    entropy_detect(model, x::Array; high_threshold::Float64 = 0.8,
                   low_threshold::Float64 = 0.1)

Detect based on softmax entropy (adversarials often have unusual entropy).
"""
function entropy_detect(model, x::Array;
                        high_threshold::Float64 = 0.8,
                        low_threshold::Float64 = 0.1)::Dict{String, Any}
    
    pred = predict(model, x)
    probs = pred.probabilities
    
    # Shannon entropy
    entropy = -sum(p * log(p + 1e-10) for p in probs if p > 0)
    
    # Normalize by max entropy
    max_entropy = log(length(probs))
    normalized_entropy = entropy / max_entropy
    
    # Score: adversarials often have very high or very low entropy
    if normalized_entropy > high_threshold || normalized_entropy < low_threshold
        score = 1.0 - abs(normalized_entropy - 0.5) * 2
    else
        score = 0.0
    end
    
    return Dict(
        "score" => 1.0 - score,  # Invert so high = adversarial
        "entropy" => entropy,
        "normalized_entropy" => normalized_entropy,
        "is_adversarial" => normalized_entropy > high_threshold || normalized_entropy < low_threshold,
        "method" => "entropy"
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              MAHALANOBIS DISTANCE
# ───────────────────────────────────────────────────────────────────────────────

"""
    mahalanobis_detect(model, x::Array; threshold::Float64 = 10.0)

Mahalanobis distance-based detection (Lee et al., 2018).
"""
function mahalanobis_detect(model, x::Array;
                            threshold::Float64 = 10.0)::Dict{String, Any}
    
    layers = get_layer_names(model)
    if isempty(layers)
        return Dict("score" => 0.5, "distance" => 0.0, "method" => "mahalanobis")
    end
    
    # Get representation
    layer = layers[end]
    rep = vec(get_layer_output(model, x, layer))
    
    # Estimate mean and covariance (simplified - would use training data)
    mean_rep = zeros(length(rep))
    cov_rep = I * 1.0
    
    # Mahalanobis distance
    diff = rep .- mean_rep
    distance = sqrt(abs(diff' * (cov_rep \ diff)))
    
    return Dict(
        "score" => min(distance / threshold, 1.0),
        "distance" => distance,
        "is_adversarial" => distance > threshold,
        "method" => "mahalanobis"
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              COMBINED DETECTOR
# ───────────────────────────────────────────────────────────────────────────────

"""
    create_detector(methods::Vector{Symbol}; weights::Union{Vector{Float64}, Nothing} = nothing)

Create an ensemble detector.
"""
function create_detector(methods::Vector{Symbol};
                         weights::Union{Vector{Float64}, Nothing} = nothing)::Defense
    
    if weights === nothing
        weights = ones(length(methods)) ./ length(methods)
    end
    
    detect_fn = (model, x) -> begin
        scores = Float64[]
        
        for (method, weight) in zip(methods, weights)
            if method == :feature_squeezing
                result = feature_squeezing_detect(model, x)
            elseif method == :lid
                result = lid_detect(model, x)
            elseif method == :input_transformation
                result = input_transformation_detect(model, x)
            elseif method == :entropy
                result = entropy_detect(model, x)
            elseif method == :mahalanobis
                result = mahalanobis_detect(model, x)
            else
                continue
            end
            
            push!(scores, result["score"] * weight)
        end
        
        aggregate = sum(scores)
        return aggregate > 0.5
    end
    
    return Defense(
        :ensemble_detector,
        Dict(:methods => methods, :weights => weights),
        detect_fn
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              DISPLAY
# ───────────────────────────────────────────────────────────────────────────────

"""
    display_detection_result(result::Dict)

Display detection results.
"""
function display_detection_result(result::Dict)
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    if result["is_adversarial"]
        println("║              ⚠️  ADVERSARIAL INPUT DETECTED                  ║")
    else
        println("║              ✓  INPUT APPEARS CLEAN                          ║")
    end
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ Aggregate Score: $(rpad(round(result["aggregate_score"], digits=4), 42))║")
    println("║ Confidence:      $(rpad(round(result["confidence"] * 100, digits=1), 39))% ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ Method Results:                                              ║")
    
    for (key, val) in result
        if isa(val, Dict) && haskey(val, "score")
            method = get(val, "method", key)
            score = round(val["score"], digits=4)
            detected = val["is_adversarial"] ? "⚠️" : "✓"
            println("║   $detected $(rpad(method, 20)) Score: $(rpad(score, 18))║")
        end
    end
    
    println("╚══════════════════════════════════════════════════════════════╝")
    println()
end
