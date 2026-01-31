# ═══════════════════════════════════════════════════════════════════════════════
#                              MIRAGE - Black-Box Attacks
# ═══════════════════════════════════════════════════════════════════════════════
# Query-based adversarial attacks without gradient access
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
#                              SQUARE ATTACK
# ───────────────────────────────────────────────────────────────────────────────

"""
    square_attack(model::Model, x::Array, y::Int; epsilon::Float64 = 0.05,
                  max_queries::Int = 5000, p_init::Float64 = 0.8)

Square Attack (Andriushchenko et al., 2020).

Query-efficient score-based attack using random square perturbations.
"""
function square_attack(model::Model, x::Array, y::Int;
                       epsilon::Float64 = 0.05,
                       max_queries::Int = 5000,
                       p_init::Float64 = 0.8)::AttackResult
    
    start_time = time()
    log_attack(:square, "input")
    
    x = Float32.(x)
    h, w = size(x)[1:2]
    c = ndims(x) > 2 ? size(x, 3) : 1
    
    original_pred = predict(model, x)
    
    # Initialize with random perturbation
    x_adv = copy(x)
    delta = zeros(Float32, size(x))
    
    # Random initialization
    init_delta = (2 * (rand(Float32, size(x)) .> 0.5) .- 1) .* epsilon
    delta = init_delta
    x_adv = clamp.(x .+ delta, 0.0f0, 1.0f0)
    
    best_adv = copy(x_adv)
    best_loss = margin_loss(model, x_adv, y)
    queries = 1
    
    for iter in 1:max_queries
        queries >= max_queries && break
        
        # Adaptive square size
        p = p_init * (1 - iter / max_queries)^0.5
        s = max(1, round(Int, sqrt(p * h * w)))
        
        # Random position
        ri = rand(1:h-s+1)
        ci = rand(1:w-s+1)
        
        # Random color change
        color_delta = (2 * (rand() > 0.5) - 1) * epsilon
        
        # Apply square perturbation
        delta_new = copy(delta)
        if ndims(x) > 2
            delta_new[ri:ri+s-1, ci:ci+s-1, :] .= color_delta
        else
            delta_new[ri:ri+s-1, ci:ci+s-1] .= color_delta
        end
        
        x_new = clamp.(x .+ delta_new, 0.0f0, 1.0f0)
        new_loss = margin_loss(model, x_new, y)
        queries += 1
        
        # Accept if loss improved
        if new_loss < best_loss
            best_loss = new_loss
            delta = delta_new
            x_adv = x_new
            best_adv = copy(x_adv)
        end
        
        # Check success
        pred = predict(model, x_adv)
        if pred.label != y
            break
        end
        
        if iter % 100 == 0
            CONFIG[].verbose && display_attack_progress(iter, max_queries, best_loss, false)
        end
    end
    
    println()
    
    final_pred = predict(model, best_adv)
    
    result = AttackResult(
        final_pred.label != original_pred.label ? SUCCESS : FAILED,
        vec(x),
        vec(best_adv),
        original_pred.label,
        final_pred.label,
        compute_norm(best_adv .- x, L2),
        compute_norm(best_adv .- x, Linf),
        queries,
        queries,
        original_pred.confidence,
        final_pred.confidence,
        time() - start_time
    )
    
    CONFIG[].verbose && display_result(result)
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
#                              HOPSKIPJUMP
# ───────────────────────────────────────────────────────────────────────────────

"""
    hopskipjump(model::Model, x::Array; target_label::Union{Int, Nothing} = nothing,
                max_queries::Int = 10000, gamma::Float64 = 1.0,
                stepsize_search::Symbol = :geometric)

HopSkipJump Attack (Chen et al., 2019).

Decision-based attack using binary search along directions.
"""
function hopskipjump(model::Model, x::Array;
                     target_label::Union{Int, Nothing} = nothing,
                     max_queries::Int = 10000,
                     gamma::Float64 = 1.0,
                     stepsize_search::Symbol = :geometric)::AttackResult
    
    start_time = time()
    log_attack(:hopskipjump, "input")
    
    x = Float32.(x)
    original_pred = predict(model, x)
    original_label = original_pred.label
    
    targeted = target_label !== nothing
    queries = 0
    
    # Initialize: find adversarial starting point
    x_adv = if targeted
        find_initial_targeted(model, x, target_label, queries)
    else
        find_initial_untargeted(model, x, original_label)
    end
    queries += 100  # Approximate initialization cost
    
    if x_adv === nothing
        log_error("Could not find initial adversarial example")
        return AttackResult(
            FAILED, vec(x), vec(x), original_label, original_label,
            0.0, 0.0, queries, 0, original_pred.confidence, original_pred.confidence,
            time() - start_time
        )
    end
    
    # Main loop
    for iter in 1:max_queries÷10
        queries >= max_queries && break
        
        # Estimate gradient direction at boundary
        direction = estimate_boundary_gradient(model, x, x_adv, original_label, targeted, target_label)
        queries += 50  # Gradient estimation cost
        
        # Geometric search for step size
        step_size = gamma
        for _ in 1:10
            x_candidate = x_adv .+ step_size .* direction
            x_candidate = clamp.(x_candidate, 0.0f0, 1.0f0)
            
            pred = predict(model, x_candidate)
            queries += 1
            
            success = targeted ? (pred.label == target_label) : (pred.label != original_label)
            
            if success
                x_adv = x_candidate
                break
            end
            
            step_size *= 0.5
        end
        
        # Binary search to boundary
        x_adv = binary_search_boundary(model, x, x_adv, original_label, targeted, target_label)
        queries += 20  # Binary search cost
        
        if iter % 10 == 0
            dist = compute_norm(x_adv .- x, L2)
            CONFIG[].verbose && display_attack_progress(iter, max_queries÷10, dist, true)
        end
    end
    
    println()
    
    final_pred = predict(model, x_adv)
    
    result = AttackResult(
        final_pred.label != original_pred.label ? SUCCESS : FAILED,
        vec(x),
        vec(x_adv),
        original_label,
        final_pred.label,
        compute_norm(x_adv .- x, L2),
        compute_norm(x_adv .- x, Linf),
        queries,
        max_queries÷10,
        original_pred.confidence,
        final_pred.confidence,
        time() - start_time
    )
    
    CONFIG[].verbose && display_result(result)
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
#                              BOUNDARY ATTACK
# ───────────────────────────────────────────────────────────────────────────────

"""
    boundary_attack(model::Model, x::Array; max_iterations::Int = 10000,
                    spherical_step::Float64 = 0.01, source_step::Float64 = 0.01,
                    step_adaptation::Float64 = 1.5)

Boundary Attack (Brendel et al., 2017).

Decision-based attack that walks along the decision boundary.
"""
function boundary_attack(model::Model, x::Array;
                         max_iterations::Int = 10000,
                         spherical_step::Float64 = 0.01,
                         source_step::Float64 = 0.01,
                         step_adaptation::Float64 = 1.5)::AttackResult
    
    start_time = time()
    log_attack(:boundary, "input")
    
    x = Float32.(x)
    original_pred = predict(model, x)
    original_label = original_pred.label
    
    # Initialize with random adversarial
    x_adv = find_initial_untargeted(model, x, original_label)
    
    if x_adv === nothing
        log_error("Could not find initial adversarial example")
        return AttackResult(
            FAILED, vec(x), vec(x), original_label, original_label,
            0.0, 0.0, 0, 0, original_pred.confidence, original_pred.confidence,
            time() - start_time
        )
    end
    
    queries = 100
    best_dist = compute_norm(x_adv .- x, L2)
    
    for iter in 1:max_iterations
        # Orthogonal perturbation (spherical step)
        perturbation = randn(Float32, size(x))
        
        # Project to orthogonal to direction x -> x_adv
        direction = x_adv .- x
        direction_norm = direction ./ (compute_norm(direction, L2) + 1e-10)
        perturbation = perturbation .- sum(perturbation .* direction_norm) .* direction_norm
        perturbation = perturbation ./ (compute_norm(perturbation, L2) + 1e-10)
        
        # Scale
        perturbation .*= spherical_step * compute_norm(x_adv .- x, L2)
        
        # Step toward source
        source_direction = (x .- x_adv) ./ (compute_norm(x .- x_adv, L2) + 1e-10)
        source_perturbation = source_step * compute_norm(x_adv .- x, L2) .* source_direction
        
        # Candidate
        x_candidate = x_adv .+ perturbation .+ source_perturbation
        x_candidate = clamp.(x_candidate, 0.0f0, 1.0f0)
        
        pred = predict(model, x_candidate)
        queries += 1
        
        if pred.label != original_label
            x_adv = x_candidate
            dist = compute_norm(x_adv .- x, L2)
            
            if dist < best_dist
                best_dist = dist
                # Increase step sizes
                spherical_step *= step_adaptation
                source_step *= step_adaptation
            end
        else
            # Decrease step sizes
            spherical_step /= step_adaptation
            source_step /= step_adaptation
        end
        
        if iter % 500 == 0
            CONFIG[].verbose && display_attack_progress(iter, max_iterations, best_dist, true)
        end
    end
    
    println()
    
    final_pred = predict(model, x_adv)
    
    result = AttackResult(
        final_pred.label != original_pred.label ? SUCCESS : FAILED,
        vec(x),
        vec(x_adv),
        original_label,
        final_pred.label,
        compute_norm(x_adv .- x, L2),
        compute_norm(x_adv .- x, Linf),
        queries,
        max_iterations,
        original_pred.confidence,
        final_pred.confidence,
        time() - start_time
    )
    
    CONFIG[].verbose && display_result(result)
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
#                              SIMBA
# ───────────────────────────────────────────────────────────────────────────────

"""
    simba(model::Model, x::Array, y::Int; epsilon::Float64 = 0.2,
          max_queries::Int = 10000, freq_dims::Int = 28)

Simple Black-box Attack (Guo et al., 2019).

Uses random sign gradients in DCT space.
"""
function simba(model::Model, x::Array, y::Int;
               epsilon::Float64 = 0.2,
               max_queries::Int = 10000,
               freq_dims::Int = 28)::AttackResult
    
    start_time = time()
    log_attack(:simba, "input")
    
    x = Float32.(x)
    original_pred = predict(model, x)
    
    x_adv = copy(x)
    delta = zeros(Float32, size(x))
    queries = 0
    
    # Generate DCT basis
    num_dims = prod(size(x))
    perm = randperm(num_dims)
    
    for idx in perm
        queries >= max_queries && break
        
        # Create basis vector
        basis = zeros(Float32, size(x))
        basis[idx] = epsilon
        
        # Try positive and negative direction
        for direction in [1, -1]
            x_candidate = clamp.(x_adv .+ direction .* basis, 0.0f0, 1.0f0)
            pred = predict(model, x_candidate)
            queries += 1
            
            if pred.label != y
                # Success
                final_pred = pred
                result = AttackResult(
                    SUCCESS,
                    vec(x),
                    vec(x_candidate),
                    original_pred.label,
                    final_pred.label,
                    compute_norm(x_candidate .- x, L2),
                    compute_norm(x_candidate .- x, Linf),
                    queries,
                    queries,
                    original_pred.confidence,
                    final_pred.confidence,
                    time() - start_time
                )
                CONFIG[].verbose && display_result(result)
                return result
            end
            
            # Check if probability decreased
            if pred.probabilities[y] < predict(model, x_adv).probabilities[y]
                x_adv = x_candidate
                break
            end
        end
        
        if queries % 500 == 0
            pred = predict(model, x_adv)
            CONFIG[].verbose && display_attack_progress(queries, max_queries, pred.probabilities[y], false)
        end
    end
    
    println()
    
    final_pred = predict(model, x_adv)
    
    result = AttackResult(
        final_pred.label != original_pred.label ? SUCCESS : FAILED,
        vec(x),
        vec(x_adv),
        original_pred.label,
        final_pred.label,
        compute_norm(x_adv .- x, L2),
        compute_norm(x_adv .- x, Linf),
        queries,
        queries,
        original_pred.confidence,
        final_pred.confidence,
        time() - start_time
    )
    
    CONFIG[].verbose && display_result(result)
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
#                              NES ATTACK
# ───────────────────────────────────────────────────────────────────────────────

"""
    nes_attack(model::Model, x::Array, y::Int; epsilon::Float64 = 0.03,
               max_queries::Int = 5000, samples::Int = 100, sigma::Float64 = 0.001,
               learning_rate::Float64 = 0.01)

Natural Evolution Strategies Attack (Ilyas et al., 2018).

Uses NES gradient estimation for black-box optimization.
"""
function nes_attack(model::Model, x::Array, y::Int;
                    epsilon::Float64 = 0.03,
                    max_queries::Int = 5000,
                    samples::Int = 100,
                    sigma::Float64 = 0.001,
                    learning_rate::Float64 = 0.01)::AttackResult
    
    start_time = time()
    log_attack(:nes, "input")
    
    x = Float32.(x)
    original_pred = predict(model, x)
    
    x_adv = copy(x)
    queries = 0
    
    iterations = max_queries ÷ (2 * samples)
    
    for iter in 1:iterations
        # Estimate gradient using NES
        grad = zeros(Float32, size(x))
        
        for _ in 1:samples
            noise = randn(Float32, size(x))
            
            x_plus = clamp.(x_adv .+ sigma .* noise, 0.0f0, 1.0f0)
            x_minus = clamp.(x_adv .- sigma .* noise, 0.0f0, 1.0f0)
            
            pred_plus = predict(model, x_plus)
            pred_minus = predict(model, x_minus)
            queries += 2
            
            # Use probability of true class as loss
            loss_plus = pred_plus.probabilities[y]
            loss_minus = pred_minus.probabilities[y]
            
            grad .+= (loss_plus - loss_minus) .* noise
        end
        
        grad ./= (2 * sigma * samples)
        
        # Update adversarial example (gradient ascent to decrease prob)
        x_adv .-= learning_rate .* sign.(grad)
        x_adv = clip_perturbation(x_adv, x, epsilon, Linf, 0.0, 1.0)
        
        # Check success
        pred = predict(model, x_adv)
        queries += 1
        
        if pred.label != y
            break
        end
        
        if iter % 10 == 0
            CONFIG[].verbose && display_attack_progress(iter, iterations, pred.probabilities[y], false)
        end
    end
    
    println()
    
    final_pred = predict(model, x_adv)
    
    result = AttackResult(
        final_pred.label != original_pred.label ? SUCCESS : FAILED,
        vec(x),
        vec(x_adv),
        original_pred.label,
        final_pred.label,
        compute_norm(x_adv .- x, L2),
        compute_norm(x_adv .- x, Linf),
        queries,
        iterations,
        original_pred.confidence,
        final_pred.confidence,
        time() - start_time
    )
    
    CONFIG[].verbose && display_result(result)
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
#                              HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

"""
    margin_loss(model::Model, x::Array, y::Int)

Compute margin loss (negative margin).
"""
function margin_loss(model::Model, x::Array, y::Int)::Float64
    pred = predict(model, x)
    probs = copy(pred.probabilities)
    
    true_prob = probs[y]
    probs[y] = 0.0
    max_other = maximum(probs)
    
    return true_prob - max_other  # Negative when misclassified
end

"""
    find_initial_untargeted(model::Model, x::Array, original_label::Int)

Find initial adversarial example for untargeted attack.
"""
function find_initial_untargeted(model::Model, x::Array, original_label::Int)::Union{Array, Nothing}
    # Try random noise
    for scale in [0.1, 0.3, 0.5, 1.0]
        for _ in 1:10
            noise = randn(Float32, size(x)) .* scale
            x_adv = clamp.(x .+ noise, 0.0f0, 1.0f0)
            
            pred = predict(model, x_adv)
            if pred.label != original_label
                return x_adv
            end
        end
    end
    
    # Try random image
    x_random = rand(Float32, size(x))
    pred = predict(model, x_random)
    if pred.label != original_label
        return x_random
    end
    
    return nothing
end

"""
    find_initial_targeted(model::Model, x::Array, target::Int, queries::Int)

Find initial adversarial example for targeted attack.
"""
function find_initial_targeted(model::Model, x::Array, target::Int, queries::Int)::Union{Array, Nothing}
    # Search for image classified as target
    for _ in 1:100
        x_random = rand(Float32, size(x))
        pred = predict(model, x_random)
        
        if pred.label == target
            return x_random
        end
    end
    
    return nothing
end

"""
    estimate_boundary_gradient(model, x, x_adv, original_label, targeted, target)

Estimate gradient at decision boundary using sampling.
"""
function estimate_boundary_gradient(model::Model, x::Array, x_adv::Array,
                                    original_label::Int, targeted::Bool,
                                    target::Union{Int, Nothing})::Array{Float32}
    
    direction = x .- x_adv
    direction ./= compute_norm(direction, L2) + 1e-10
    
    grad = zeros(Float32, size(x))
    num_samples = 50
    delta = 0.01
    
    for _ in 1:num_samples
        noise = randn(Float32, size(x))
        noise .-= sum(noise .* direction) .* direction  # Orthogonalize
        noise ./= compute_norm(noise, L2) + 1e-10
        
        x_plus = x_adv .+ delta .* noise
        x_minus = x_adv .- delta .* noise
        
        pred_plus = predict(model, clamp.(x_plus, 0.0f0, 1.0f0))
        pred_minus = predict(model, clamp.(x_minus, 0.0f0, 1.0f0))
        
        # Binary indicator
        if targeted
            success_plus = pred_plus.label == target ? 1.0 : 0.0
            success_minus = pred_minus.label == target ? 1.0 : 0.0
        else
            success_plus = pred_plus.label != original_label ? 1.0 : 0.0
            success_minus = pred_minus.label != original_label ? 1.0 : 0.0
        end
        
        grad .+= (success_plus - success_minus) .* noise
    end
    
    grad ./= num_samples
    return grad
end

"""
    binary_search_boundary(model, x, x_adv, original_label, targeted, target)

Binary search to find point on decision boundary.
"""
function binary_search_boundary(model::Model, x::Array, x_adv::Array,
                                original_label::Int, targeted::Bool,
                                target::Union{Int, Nothing})::Array{Float32}
    
    low = 0.0f0
    high = 1.0f0
    
    for _ in 1:20
        mid = (low + high) / 2
        x_mid = (1 - mid) .* x_adv .+ mid .* x
        x_mid = clamp.(x_mid, 0.0f0, 1.0f0)
        
        pred = predict(model, x_mid)
        
        if targeted
            if pred.label == target
                low = mid  # Move toward source
            else
                high = mid
            end
        else
            if pred.label != original_label
                low = mid  # Move toward source
            else
                high = mid
            end
        end
    end
    
    return clamp.((1 - low) .* x_adv .+ low .* x, 0.0f0, 1.0f0)
end
