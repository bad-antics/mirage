# ═══════════════════════════════════════════════════════════════════════════════
#                              MIRAGE - Decision Boundary Analysis
# ═══════════════════════════════════════════════════════════════════════════════
# Tools for exploring and mapping model decision boundaries
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
#                              BOUNDARY MAPPING
# ───────────────────────────────────────────────────────────────────────────────

"""
    map_decision_boundary(model, x::Array; direction1::Union{Array, Nothing} = nothing,
                          direction2::Union{Array, Nothing} = nothing,
                          range::Float64 = 0.5, resolution::Int = 50)

Map the decision boundary around an input in 2D.
"""
function map_decision_boundary(model, x::Array;
                               direction1::Union{Array, Nothing} = nothing,
                               direction2::Union{Array, Nothing} = nothing,
                               range::Float64 = 0.5,
                               resolution::Int = 50)::Dict{String, Any}
    
    log_info("Mapping decision boundary ($(resolution)×$(resolution) grid)...")
    
    # Get base prediction
    base_pred = predict(model, x)
    base_class = base_pred.label
    
    # Default directions: random orthogonal vectors
    if direction1 === nothing
        direction1 = randn(size(x))
        direction1 ./= norm(direction1)
    end
    
    if direction2 === nothing
        direction2 = randn(size(x))
        # Make orthogonal to direction1
        direction2 .-= dot(direction2, direction1) .* direction1
        direction2 ./= norm(direction2)
    end
    
    # Create grid
    alphas = range(-range, range, length=resolution)
    betas = range(-range, range, length=resolution)
    
    predictions = zeros(Int, resolution, resolution)
    confidences = zeros(Float64, resolution, resolution)
    
    for (i, alpha) in enumerate(alphas)
        for (j, beta) in enumerate(betas)
            # Perturb input
            x_perturbed = x .+ alpha .* direction1 .+ beta .* direction2
            
            # Get prediction
            pred = predict(model, x_perturbed)
            predictions[i, j] = pred.label
            confidences[i, j] = pred.confidence
        end
    end
    
    # Find boundary points
    boundary_points = find_boundary_points_grid(predictions, alphas, betas)
    
    log_success("Found $(length(boundary_points)) boundary points")
    
    return Dict(
        "predictions" => predictions,
        "confidences" => confidences,
        "alphas" => collect(alphas),
        "betas" => collect(betas),
        "direction1" => direction1,
        "direction2" => direction2,
        "base_class" => base_class,
        "boundary_points" => boundary_points
    )
end

"""Find boundary points in prediction grid."""
function find_boundary_points_grid(predictions::Matrix{Int}, alphas::AbstractRange,
                                    betas::AbstractRange)::Vector{BoundaryPoint}
    
    points = BoundaryPoint[]
    n, m = size(predictions)
    
    for i in 1:n-1
        for j in 1:m-1
            # Check if this cell is on boundary
            neighbors = [predictions[i, j], predictions[i+1, j], 
                        predictions[i, j+1], predictions[i+1, j+1]]
            
            if length(unique(neighbors)) > 1
                # This is a boundary cell
                alpha = (alphas[i] + alphas[i+1]) / 2
                beta = (betas[j] + betas[j+1]) / 2
                
                push!(points, BoundaryPoint(
                    (alpha, beta),
                    predictions[i, j],
                    predictions[i+1, j+1],
                    0.0  # Distance computed later
                ))
            end
        end
    end
    
    return points
end

# ───────────────────────────────────────────────────────────────────────────────
#                              BOUNDARY DISTANCE
# ───────────────────────────────────────────────────────────────────────────────

"""
    distance_to_boundary(model, x::Array; direction::Union{Array, Nothing} = nothing,
                         max_distance::Float64 = 1.0, tolerance::Float64 = 1e-4)

Find distance to nearest decision boundary.
"""
function distance_to_boundary(model, x::Array;
                              direction::Union{Array, Nothing} = nothing,
                              max_distance::Float64 = 1.0,
                              tolerance::Float64 = 1e-4)::Float64
    
    base_pred = predict(model, x)
    base_class = base_pred.label
    
    # Default: use gradient direction
    if direction === nothing
        grad_info = get_gradient(model, x, base_class)
        direction = -grad_info.gradient  # Move away from class
        direction ./= (norm(direction) + 1e-10)
    end
    
    # Binary search for boundary
    low, high = 0.0, max_distance
    
    while high - low > tolerance
        mid = (low + high) / 2
        x_test = x .+ mid .* direction
        
        pred = predict(model, x_test)
        
        if pred.label == base_class
            low = mid
        else
            high = mid
        end
    end
    
    return (low + high) / 2
end

"""
    minimum_boundary_distance(model, x::Array; n_directions::Int = 100)

Find minimum distance to boundary across multiple directions.
"""
function minimum_boundary_distance(model, x::Array; n_directions::Int = 100)::Dict{String, Any}
    log_info("Searching for minimum boundary distance ($n_directions directions)...")
    
    min_distance = Inf
    min_direction = nothing
    all_distances = Float64[]
    
    for _ in 1:n_directions
        # Random direction
        direction = randn(size(x))
        direction ./= norm(direction)
        
        dist = distance_to_boundary(model, x, direction=direction)
        push!(all_distances, dist)
        
        if dist < min_distance
            min_distance = dist
            min_direction = direction
        end
    end
    
    log_success("Minimum distance: $(round(min_distance, digits=4))")
    
    return Dict(
        "min_distance" => min_distance,
        "min_direction" => min_direction,
        "mean_distance" => mean(all_distances),
        "std_distance" => std(all_distances),
        "all_distances" => all_distances
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              BOUNDARY CURVATURE
# ───────────────────────────────────────────────────────────────────────────────

"""
    boundary_curvature(model, x::Array; n_samples::Int = 20, epsilon::Float64 = 0.1)

Estimate local curvature of decision boundary.
"""
function boundary_curvature(model, x::Array;
                            n_samples::Int = 20,
                            epsilon::Float64 = 0.1)::Dict{String, Float64}
    
    log_info("Estimating boundary curvature...")
    
    # Find nearest boundary point
    boundary_result = minimum_boundary_distance(model, x, n_directions=50)
    boundary_dist = boundary_result["min_distance"]
    boundary_dir = boundary_result["min_direction"]
    
    # Point on boundary
    x_boundary = x .+ boundary_dist .* boundary_dir
    
    # Sample points around boundary
    tangent_directions = []
    curvatures = Float64[]
    
    for _ in 1:n_samples
        # Random tangent direction (orthogonal to boundary_dir)
        tangent = randn(size(x))
        tangent .-= dot(tangent, boundary_dir) .* boundary_dir
        tangent ./= (norm(tangent) + 1e-10)
        
        # Check boundary at nearby points
        dist_plus = distance_to_boundary(model, x_boundary .+ epsilon .* tangent)
        dist_minus = distance_to_boundary(model, x_boundary .- epsilon .* tangent)
        
        # Second derivative estimate (curvature)
        curvature = abs(dist_plus + dist_minus - 2 * 0) / (epsilon^2)
        push!(curvatures, curvature)
    end
    
    mean_curv = mean(curvatures)
    max_curv = maximum(curvatures)
    
    log_success("Mean curvature: $(round(mean_curv, digits=4))")
    
    return Dict(
        "mean_curvature" => mean_curv,
        "max_curvature" => max_curv,
        "min_curvature" => minimum(curvatures),
        "boundary_distance" => boundary_dist
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              BOUNDARY WALK
# ───────────────────────────────────────────────────────────────────────────────

"""
    walk_boundary(model, x::Array; steps::Int = 100, step_size::Float64 = 0.01)

Walk along the decision boundary.
"""
function walk_boundary(model, x::Array;
                       steps::Int = 100,
                       step_size::Float64 = 0.01)::Vector{Array}
    
    log_info("Walking decision boundary ($steps steps)...")
    
    base_pred = predict(model, x)
    base_class = base_pred.label
    
    # Find initial boundary point
    boundary_result = minimum_boundary_distance(model, x)
    boundary_dir = boundary_result["min_direction"]
    boundary_dist = boundary_result["min_distance"]
    
    current = x .+ boundary_dist .* boundary_dir
    path = [copy(current)]
    
    # Random tangent direction
    tangent = randn(size(x))
    tangent .-= dot(tangent, boundary_dir) .* boundary_dir
    tangent ./= (norm(tangent) + 1e-10)
    
    for step in 1:steps
        # Move along tangent
        next = current .+ step_size .* tangent
        
        # Project back to boundary
        pred = predict(model, next)
        if pred.label == base_class
            # Move outward
            for _ in 1:10
                next .+= 0.001 .* boundary_dir
                pred = predict(model, next)
                pred.label != base_class && break
            end
        else
            # Move inward
            for _ in 1:10
                next .-= 0.001 .* boundary_dir
                pred = predict(model, next)
                pred.label == base_class && break
            end
            next .+= 0.001 .* boundary_dir  # Back to boundary
        end
        
        current = next
        push!(path, copy(current))
        
        # Update tangent (follow boundary)
        new_grad = get_gradient(model, current, base_class).gradient
        new_normal = -new_grad ./ (norm(new_grad) + 1e-10)
        tangent .-= dot(tangent, new_normal) .* new_normal
        tangent ./= (norm(tangent) + 1e-10)
    end
    
    log_success("Boundary walk complete")
    
    return path
end

# ───────────────────────────────────────────────────────────────────────────────
#                              ADVERSARIAL SUBSPACES
# ───────────────────────────────────────────────────────────────────────────────

"""
    find_adversarial_subspace(model, x::Array; n_components::Int = 10)

Find low-dimensional subspace containing adversarial directions.
"""
function find_adversarial_subspace(model, x::Array;
                                   n_components::Int = 10)::Dict{String, Any}
    
    log_info("Finding $n_components-dimensional adversarial subspace...")
    
    base_pred = predict(model, x)
    base_class = base_pred.label
    
    # Collect successful adversarial directions
    adversarial_dirs = []
    
    for _ in 1:n_components * 10
        # Random direction
        direction = randn(size(x))
        direction ./= norm(direction)
        
        # Check if adversarial
        for scale in [0.01, 0.05, 0.1, 0.2, 0.3]
            x_adv = x .+ scale .* direction
            pred = predict(model, x_adv)
            
            if pred.label != base_class
                push!(adversarial_dirs, direction .* scale)
                break
            end
        end
        
        length(adversarial_dirs) >= n_components * 3 && break
    end
    
    if isempty(adversarial_dirs)
        log_warning("No adversarial directions found")
        return Dict("components" => [], "variances" => [])
    end
    
    # PCA on adversarial directions
    D = reduce(hcat, [vec(d) for d in adversarial_dirs])'
    D_centered = D .- mean(D, dims=1)
    
    # SVD for PCA
    U, S, V = svd(D_centered)
    
    # Top components
    n_keep = min(n_components, length(S))
    components = [reshape(V[:, i], size(x)) for i in 1:n_keep]
    variances = S[1:n_keep].^2 ./ sum(S.^2)
    
    log_success("Found $n_keep principal adversarial directions")
    
    return Dict(
        "components" => components,
        "variances" => variances,
        "explained_variance" => sum(variances)
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              ANALYSIS HELPERS
# ───────────────────────────────────────────────────────────────────────────────

"""
    robustness_profile(model, x::Array; n_samples::Int = 1000)

Compute robustness profile at different perturbation magnitudes.
"""
function robustness_profile(model, x::Array;
                            n_samples::Int = 1000)::Dict{String, Vector{Float64}}
    
    log_info("Computing robustness profile...")
    
    base_pred = predict(model, x)
    base_class = base_pred.label
    
    epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    accuracies = Float64[]
    confidences = Float64[]
    
    for eps in epsilons
        correct = 0
        total_conf = 0.0
        
        for _ in 1:n_samples
            # Random perturbation
            delta = randn(size(x))
            delta .*= eps / norm(delta)
            
            pred = predict(model, x .+ delta)
            
            if pred.label == base_class
                correct += 1
            end
            total_conf += pred.confidence
        end
        
        push!(accuracies, correct / n_samples)
        push!(confidences, total_conf / n_samples)
    end
    
    return Dict(
        "epsilons" => epsilons,
        "accuracies" => accuracies,
        "confidences" => confidences
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              DISPLAY
# ───────────────────────────────────────────────────────────────────────────────

"""
    display_boundary_map(result::Dict)

Display decision boundary map.
"""
function display_boundary_map(result::Dict)
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                    DECISION BOUNDARY MAP                     ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ Base Class:     $(rpad(result["base_class"], 43))║")
    println("║ Grid Size:      $(rpad(string(size(result["predictions"])), 43))║")
    println("║ Boundary Points:$(rpad(length(result["boundary_points"]), 43))║")
    println("╠══════════════════════════════════════════════════════════════╣")
    
    # ASCII visualization
    println("║ Prediction Map (class colors):                               ║")
    
    predictions = result["predictions"]
    classes = unique(predictions)
    class_chars = Dict(c => Char('A' + i - 1) for (i, c) in enumerate(classes))
    
    n = min(size(predictions, 1), 20)
    m = min(size(predictions, 2), 50)
    
    step_i = size(predictions, 1) ÷ n
    step_j = size(predictions, 2) ÷ m
    
    for i in 1:n
        row = "║ "
        for j in 1:m
            row *= string(class_chars[predictions[i*step_i, j*step_j]])
        end
        row *= rpad("", 60 - length(row)) * "║"
        println(row)
    end
    
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ Legend:                                                      ║")
    for (c, char) in class_chars
        println("║   $(char) = Class $c$(repeat(" ", 48))║")
    end
    
    println("╚══════════════════════════════════════════════════════════════╝")
    println()
end

"""
    display_robustness_profile(profile::Dict)

Display robustness profile.
"""
function display_robustness_profile(profile::Dict)
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                    ROBUSTNESS PROFILE                        ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ ε         Accuracy    Confidence                             ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    
    for (i, eps) in enumerate(profile["epsilons"])
        acc = round(profile["accuracies"][i] * 100, digits=1)
        conf = round(profile["confidences"][i], digits=3)
        
        bar_width = round(Int, profile["accuracies"][i] * 30)
        bar = repeat("█", bar_width) * repeat("░", 30 - bar_width)
        
        println("║ $(rpad(eps, 9)) $(rpad(acc, 10))% $(bar) ║")
    end
    
    println("╚══════════════════════════════════════════════════════════════╝")
    println()
end
