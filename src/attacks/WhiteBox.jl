# ═══════════════════════════════════════════════════════════════════════════════
#                              MIRAGE - White-Box Attacks
# ═══════════════════════════════════════════════════════════════════════════════
# Gradient-based adversarial attacks requiring model access
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
#                              FGSM
# ───────────────────────────────────────────────────────────────────────────────

"""
    fgsm(model::Model, x::Array, y::Int; epsilon::Float64 = 0.03, 
         targeted::Bool = false, target::Union{Int, Nothing} = nothing)

Fast Gradient Sign Method (Goodfellow et al., 2014).

Single-step attack that perturbs in the direction of the gradient sign.
"""
function fgsm(model::Model, x::Array, y::Int;
              epsilon::Float64 = 0.03,
              targeted::Bool = false,
              target::Union{Int, Nothing} = nothing)::AttackResult
    
    start_time = time()
    log_attack(:fgsm, "input")
    
    x = Float32.(x)
    original_pred = predict(model, x)
    
    # Compute gradient of loss w.r.t. input
    grad = compute_gradient(model, x, targeted ? target : y)
    
    # Sign of gradient
    if targeted
        # Move toward target class
        perturbation = -epsilon .* sign.(grad)
    else
        # Move away from true class
        perturbation = epsilon .* sign.(grad)
    end
    
    # Apply perturbation
    x_adv = clamp.(x .+ perturbation, 0.0f0, 1.0f0)
    
    adv_pred = predict(model, x_adv)
    
    result = AttackResult(
        adv_pred.label != original_pred.label ? SUCCESS : FAILED,
        vec(x),
        vec(x_adv),
        original_pred.label,
        adv_pred.label,
        compute_norm(x_adv .- x, L2),
        compute_norm(x_adv .- x, Linf),
        1,  # queries
        1,  # iterations
        original_pred.confidence,
        adv_pred.confidence,
        time() - start_time
    )
    
    CONFIG[].verbose && display_result(result)
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
#                              PGD
# ───────────────────────────────────────────────────────────────────────────────

"""
    pgd(model::Model, x::Array, y::Int; epsilon::Float64 = 0.03,
        alpha::Float64 = 0.01, iterations::Int = 40, random_start::Bool = true,
        norm::NormType = Linf, targeted::Bool = false, target::Union{Int, Nothing} = nothing)

Projected Gradient Descent (Madry et al., 2017).

Multi-step iterative attack with projection to epsilon ball.
"""
function pgd(model::Model, x::Array, y::Int;
             epsilon::Float64 = 0.03,
             alpha::Float64 = 0.01,
             iterations::Int = 40,
             random_start::Bool = true,
             norm::NormType = Linf,
             targeted::Bool = false,
             target::Union{Int, Nothing} = nothing)::AttackResult
    
    start_time = time()
    log_attack(:pgd, "input")
    
    x = Float32.(x)
    original_pred = predict(model, x)
    
    # Random start within epsilon ball
    if random_start
        x_adv = x .+ random_uniform_ball(size(x), epsilon, norm)
        x_adv = clamp.(x_adv, 0.0f0, 1.0f0)
    else
        x_adv = copy(x)
    end
    
    attack_target = targeted ? target : y
    best_adv = copy(x_adv)
    best_loss = Inf
    
    for iter in 1:iterations
        # Compute gradient
        grad = compute_gradient(model, x_adv, attack_target)
        
        # Compute step direction
        if norm == Linf
            step = alpha .* sign.(grad)
        elseif norm == L2
            grad_norm = compute_norm(grad, L2)
            step = alpha .* grad ./ (grad_norm + 1e-10)
        else
            step = alpha .* grad
        end
        
        # Update
        if targeted
            x_adv = x_adv .- step  # Gradient descent toward target
        else
            x_adv = x_adv .+ step  # Gradient ascent away from true class
        end
        
        # Project back to epsilon ball and valid range
        x_adv = clip_perturbation(x_adv, x, epsilon, norm, 0.0, 1.0)
        
        # Check success
        adv_pred = predict(model, x_adv)
        success = targeted ? (adv_pred.label == target) : (adv_pred.label != y)
        
        # Track best
        loss = cross_entropy_loss(adv_pred.logits !== nothing ? adv_pred.logits : adv_pred.probabilities, attack_target)
        if loss < best_loss
            best_loss = loss
            best_adv = copy(x_adv)
        end
        
        CONFIG[].verbose && display_attack_progress(iter, iterations, loss, success)
        
        if success
            println()  # Newline after progress
            break
        end
    end
    
    println()  # Newline after progress
    
    final_pred = predict(model, best_adv)
    
    result = AttackResult(
        final_pred.label != original_pred.label ? SUCCESS : FAILED,
        vec(x),
        vec(best_adv),
        original_pred.label,
        final_pred.label,
        compute_norm(best_adv .- x, L2),
        compute_norm(best_adv .- x, Linf),
        iterations,
        iterations,
        original_pred.confidence,
        final_pred.confidence,
        time() - start_time
    )
    
    CONFIG[].verbose && display_result(result)
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
#                              CARLINI-WAGNER
# ───────────────────────────────────────────────────────────────────────────────

"""
    carlini_wagner(model::Model, x::Array, y::Int; confidence::Float64 = 0.0,
                   learning_rate::Float64 = 0.01, max_iterations::Int = 1000,
                   binary_search_steps::Int = 9, initial_const::Float64 = 0.001,
                   norm::NormType = L2, targeted::Bool = false, 
                   target::Union{Int, Nothing} = nothing)

Carlini-Wagner Attack (Carlini & Wagner, 2016).

Optimization-based attack that minimizes perturbation while ensuring misclassification.
"""
function carlini_wagner(model::Model, x::Array, y::Int;
                        confidence::Float64 = 0.0,
                        learning_rate::Float64 = 0.01,
                        max_iterations::Int = 1000,
                        binary_search_steps::Int = 9,
                        initial_const::Float64 = 0.001,
                        norm::NormType = L2,
                        targeted::Bool = false,
                        target::Union{Int, Nothing} = nothing)::AttackResult
    
    start_time = time()
    log_attack(:cw, "input")
    
    x = Float32.(x)
    original_pred = predict(model, x)
    attack_target = targeted ? target : y
    
    # Transform to tanh space for unconstrained optimization
    x_tanh = atanh.(clamp.(x .* 2 .- 1, -0.999f0, 0.999f0))
    
    # Binary search for optimal c
    c_low = 0.0
    c_high = 1e10
    c = initial_const
    
    best_adv = copy(x)
    best_dist = Inf
    
    for search_step in 1:binary_search_steps
        # Initialize perturbation
        w = zeros(Float32, size(x))
        
        for iter in 1:max_iterations
            # Compute adversarial in image space
            x_adv = (tanh.(x_tanh .+ w) .+ 1) ./ 2
            
            # Compute losses
            pred = predict(model, x_adv)
            logits = pred.logits !== nothing ? pred.logits : log.(pred.probabilities .+ 1e-10)
            
            # Distance loss
            if norm == L2
                dist_loss = sum((x_adv .- x).^2)
            else
                dist_loss = maximum(abs.(x_adv .- x))
            end
            
            # Adversarial loss
            adv_loss = cw_loss(logits, attack_target, targeted, confidence)
            
            # Total loss
            total_loss = dist_loss + c * adv_loss
            
            # Compute gradient (numerical for simplicity)
            grad = numerical_gradient(w -> begin
                x_temp = (tanh.(x_tanh .+ w) .+ 1) ./ 2
                pred_temp = predict(model, x_temp)
                logits_temp = pred_temp.logits !== nothing ? pred_temp.logits : log.(pred_temp.probabilities .+ 1e-10)
                d = norm == L2 ? sum((x_temp .- x).^2) : maximum(abs.(x_temp .- x))
                a = cw_loss(logits_temp, attack_target, targeted, confidence)
                d + c * a
            end, w, h = 1e-3)
            
            # Update
            w .-= learning_rate .* grad
            
            # Check success
            success = targeted ? (pred.label == target) : (pred.label != y)
            
            if success && dist_loss < best_dist
                best_dist = dist_loss
                best_adv = copy(x_adv)
            end
            
            if iter % 100 == 0
                CONFIG[].verbose && display_attack_progress(iter, max_iterations, Float64(total_loss), success)
            end
        end
        
        println()
        
        # Adjust c based on success
        if best_dist < Inf
            c_high = min(c_high, c)
            c = (c_low + c_high) / 2
        else
            c_low = max(c_low, c)
            c = (c_low + c_high) / 2
        end
    end
    
    final_pred = predict(model, best_adv)
    
    result = AttackResult(
        final_pred.label != original_pred.label ? SUCCESS : FAILED,
        vec(x),
        vec(best_adv),
        original_pred.label,
        final_pred.label,
        compute_norm(best_adv .- x, L2),
        compute_norm(best_adv .- x, Linf),
        max_iterations * binary_search_steps,
        max_iterations * binary_search_steps,
        original_pred.confidence,
        final_pred.confidence,
        time() - start_time
    )
    
    CONFIG[].verbose && display_result(result)
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
#                              DEEPFOOL
# ───────────────────────────────────────────────────────────────────────────────

"""
    deepfool(model::Model, x::Array; max_iterations::Int = 50, 
             overshoot::Float64 = 0.02, num_classes::Int = 10)

DeepFool Attack (Moosavi-Dezfooli et al., 2016).

Finds minimal perturbation by iteratively moving toward decision boundary.
"""
function deepfool(model::Model, x::Array;
                  max_iterations::Int = 50,
                  overshoot::Float64 = 0.02,
                  num_classes::Int = 10)::AttackResult
    
    start_time = time()
    log_attack(:deepfool, "input")
    
    x = Float32.(x)
    x_adv = copy(x)
    original_pred = predict(model, x)
    original_label = original_pred.label
    
    total_perturbation = zeros(Float32, size(x))
    
    for iter in 1:max_iterations
        pred = predict(model, x_adv)
        
        if pred.label != original_label
            break
        end
        
        # Compute gradients for all classes
        gradients = Dict{Int, Array{Float32}}()
        for k in 1:num_classes
            gradients[k] = compute_gradient(model, x_adv, k)
        end
        
        grad_orig = gradients[original_label]
        
        # Find closest boundary
        min_dist = Inf
        optimal_perturbation = zeros(Float32, size(x))
        
        for k in 1:num_classes
            k == original_label && continue
            
            w_k = gradients[k] .- grad_orig
            f_k = pred.probabilities[k] - pred.probabilities[original_label]
            
            dist = abs(f_k) / (compute_norm(w_k, L2) + 1e-10)
            
            if dist < min_dist
                min_dist = dist
                optimal_perturbation = (abs(f_k) / (compute_norm(w_k, L2)^2 + 1e-10)) .* w_k
            end
        end
        
        # Update
        x_adv .+= (1 + overshoot) .* optimal_perturbation
        x_adv = clamp.(x_adv, 0.0f0, 1.0f0)
        total_perturbation .+= optimal_perturbation
        
        CONFIG[].verbose && display_attack_progress(iter, max_iterations, min_dist, false)
    end
    
    println()
    
    final_pred = predict(model, x_adv)
    
    result = AttackResult(
        final_pred.label != original_label ? SUCCESS : FAILED,
        vec(x),
        vec(x_adv),
        original_label,
        final_pred.label,
        compute_norm(x_adv .- x, L2),
        compute_norm(x_adv .- x, Linf),
        max_iterations,
        max_iterations,
        original_pred.confidence,
        final_pred.confidence,
        time() - start_time
    )
    
    CONFIG[].verbose && display_result(result)
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
#                              AUTO-PGD
# ───────────────────────────────────────────────────────────────────────────────

"""
    apgd(model::Model, x::Array, y::Int; epsilon::Float64 = 0.03,
         iterations::Int = 100, restarts::Int = 1, norm::NormType = Linf,
         loss::Symbol = :ce)

Auto-PGD (Croce & Hein, 2020).

PGD with automatic step size adaptation.
"""
function apgd(model::Model, x::Array, y::Int;
              epsilon::Float64 = 0.03,
              iterations::Int = 100,
              restarts::Int = 1,
              norm::NormType = Linf,
              loss::Symbol = :ce)::AttackResult
    
    start_time = time()
    log_attack(:apgd, "input")
    
    x = Float32.(x)
    original_pred = predict(model, x)
    
    best_adv = copy(x)
    best_loss = -Inf
    
    for restart in 1:restarts
        # Random start
        x_adv = x .+ random_uniform_ball(size(x), epsilon, norm)
        x_adv = clamp.(x_adv, 0.0f0, 1.0f0)
        
        # Adaptive step size
        alpha = 2 * epsilon / iterations
        
        # Checkpoints for step size reduction
        checkpoints = [0.22, 0.50, 0.75, 0.90]
        checkpoint_idx = 1
        
        prev_loss = -Inf
        loss_increased = 0
        
        for iter in 1:iterations
            grad = compute_gradient(model, x_adv, y)
            pred = predict(model, x_adv)
            
            # Compute loss
            logits = pred.logits !== nothing ? pred.logits : log.(pred.probabilities .+ 1e-10)
            current_loss = if loss == :ce
                -cross_entropy_loss(logits, y)  # Negative for maximization
            else  # :dlr
                -dlr_loss(logits, y)
            end
            
            # Check if loss increased
            if current_loss > prev_loss
                loss_increased += 1
            end
            prev_loss = current_loss
            
            # Step size adaptation at checkpoints
            progress = iter / iterations
            if checkpoint_idx <= length(checkpoints) && progress >= checkpoints[checkpoint_idx]
                if loss_increased < 0.75 * iter * (checkpoints[checkpoint_idx] - (checkpoint_idx > 1 ? checkpoints[checkpoint_idx-1] : 0))
                    alpha /= 2
                end
                checkpoint_idx += 1
            end
            
            # Update
            if norm == Linf
                x_adv .+= alpha .* sign.(grad)
            else
                grad_norm = compute_norm(grad, L2)
                x_adv .+= alpha .* grad ./ (grad_norm + 1e-10)
            end
            
            x_adv = clip_perturbation(x_adv, x, epsilon, norm, 0.0, 1.0)
            
            # Track best
            if current_loss > best_loss
                best_loss = current_loss
                best_adv = copy(x_adv)
            end
        end
    end
    
    final_pred = predict(model, best_adv)
    
    result = AttackResult(
        final_pred.label != original_pred.label ? SUCCESS : FAILED,
        vec(x),
        vec(best_adv),
        original_pred.label,
        final_pred.label,
        compute_norm(best_adv .- x, L2),
        compute_norm(best_adv .- x, Linf),
        iterations * restarts,
        iterations * restarts,
        original_pred.confidence,
        final_pred.confidence,
        time() - start_time
    )
    
    CONFIG[].verbose && display_result(result)
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
#                              AUTO-ATTACK
# ───────────────────────────────────────────────────────────────────────────────

"""
    auto_attack(model::Model, x::Array, y::Int; epsilon::Float64 = 8/255,
                norm::NormType = Linf, attacks::Vector{Symbol} = [:apgd_ce, :apgd_dlr, :fab, :square])

AutoAttack (Croce & Hein, 2020).

Ensemble of parameter-free attacks.
"""
function auto_attack(model::Model, x::Array, y::Int;
                     epsilon::Float64 = 8/255,
                     norm::NormType = Linf,
                     attacks::Vector{Symbol} = [:apgd_ce, :apgd_dlr, :fab, :square])::AttackResult
    
    start_time = time()
    log_attack(:autoattack, "input")
    
    x = Float32.(x)
    original_pred = predict(model, x)
    
    for attack in attacks
        log_info("Running $attack...")
        
        result = if attack == :apgd_ce
            apgd(model, x, y, epsilon=epsilon, norm=norm, loss=:ce)
        elseif attack == :apgd_dlr
            apgd(model, x, y, epsilon=epsilon, norm=norm, loss=:dlr)
        elseif attack == :fab
            fab_attack(model, x, y, epsilon=epsilon, norm=norm)
        elseif attack == :square
            square_attack(model, x, y, epsilon=epsilon, max_queries=5000)
        else
            continue
        end
        
        if result.status == SUCCESS
            return result
        end
    end
    
    # All attacks failed
    final_pred = predict(model, x)
    
    return AttackResult(
        FAILED,
        vec(x),
        vec(x),
        original_pred.label,
        final_pred.label,
        0.0,
        0.0,
        0,
        0,
        original_pred.confidence,
        final_pred.confidence,
        time() - start_time
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              FAB ATTACK
# ───────────────────────────────────────────────────────────────────────────────

"""
    fab_attack(model::Model, x::Array, y::Int; epsilon::Float64 = 0.03,
               iterations::Int = 100, restarts::Int = 1, norm::NormType = Linf)

Fast Adaptive Boundary Attack (Croce & Hein, 2020).

Minimizes perturbation norm while ensuring misclassification.
"""
function fab_attack(model::Model, x::Array, y::Int;
                    epsilon::Float64 = 0.03,
                    iterations::Int = 100,
                    restarts::Int = 1,
                    norm::NormType = Linf)::AttackResult
    
    start_time = time()
    x = Float32.(x)
    original_pred = predict(model, x)
    
    best_adv = copy(x)
    best_norm = Inf
    
    for _ in 1:restarts
        # Start from random point outside epsilon ball that misclassifies
        x_adv = x .+ random_uniform_ball(size(x), epsilon * 2, norm)
        x_adv = clamp.(x_adv, 0.0f0, 1.0f0)
        
        for iter in 1:iterations
            pred = predict(model, x_adv)
            
            if pred.label == y
                # Move away from decision boundary
                grad = compute_gradient(model, x_adv, y)
                step = 0.01 * sign.(grad)
                x_adv .+= step
            else
                # Project toward original while staying misclassified
                direction = x .- x_adv
                step_size = 0.01
                
                x_test = x_adv .+ step_size .* direction
                x_test = clamp.(x_test, 0.0f0, 1.0f0)
                
                test_pred = predict(model, x_test)
                if test_pred.label != y
                    x_adv = x_test
                end
            end
            
            x_adv = clamp.(x_adv, 0.0f0, 1.0f0)
            
            # Check if within epsilon and misclassifies
            current_norm = compute_norm(x_adv .- x, norm)
            pred = predict(model, x_adv)
            
            if pred.label != y && current_norm < best_norm && current_norm <= epsilon
                best_norm = current_norm
                best_adv = copy(x_adv)
            end
        end
    end
    
    final_pred = predict(model, best_adv)
    
    return AttackResult(
        final_pred.label != original_pred.label ? SUCCESS : FAILED,
        vec(x),
        vec(best_adv),
        original_pred.label,
        final_pred.label,
        compute_norm(best_adv .- x, L2),
        compute_norm(best_adv .- x, Linf),
        iterations * restarts,
        iterations * restarts,
        original_pred.confidence,
        final_pred.confidence,
        time() - start_time
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              HELPER: GRADIENT COMPUTATION
# ───────────────────────────────────────────────────────────────────────────────

"""
    compute_gradient(model::Model, x::Array, target::Int)

Compute gradient of loss w.r.t. input.
For models without analytical gradients, uses numerical estimation.
"""
function compute_gradient(model::Model, x::Array, target::Int)::Array{Float32}
    # For local models with gradient support, use analytical gradients
    # For remote models or without support, use numerical estimation
    
    if model isa LocalModel && haskey(model.weights, "gradients_enabled")
        # Analytical gradient (would use autodiff in real implementation)
        return analytical_gradient(model, x, target)
    else
        # Numerical gradient estimation
        h = 1e-4
        grad = zeros(Float32, size(x))
        
        f(x_input) = begin
            pred = predict(model, x_input)
            logits = pred.logits !== nothing ? pred.logits : log.(pred.probabilities .+ 1e-10)
            cross_entropy_loss(logits, target)
        end
        
        return Float32.(numerical_gradient(f, x, h=h))
    end
end

"""Placeholder for analytical gradient computation."""
function analytical_gradient(model::LocalModel, x::Array, target::Int)::Array{Float32}
    # In a real implementation, this would use autodiff (e.g., Zygote.jl)
    # For now, fall back to numerical
    h = 1e-4
    f(x_input) = begin
        pred = predict(model, x_input)
        logits = pred.logits !== nothing ? pred.logits : log.(pred.probabilities .+ 1e-10)
        cross_entropy_loss(logits, target)
    end
    return Float32.(numerical_gradient(f, x, h=h))
end
