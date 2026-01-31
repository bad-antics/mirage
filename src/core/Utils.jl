# ═══════════════════════════════════════════════════════════════════════════════
#                              MIRAGE - Utilities
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
#                              NORM FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

"""
    compute_norm(x::Array, norm::NormType)

Compute perturbation norm.
"""
function compute_norm(x::Array, norm::NormType)::Float64
    x_flat = vec(x)
    
    if norm == L0
        return Float64(count(!iszero, x_flat))
    elseif norm == L1
        return sum(abs, x_flat)
    elseif norm == L2
        return sqrt(sum(abs2, x_flat))
    elseif norm == Linf
        return maximum(abs, x_flat)
    end
end

"""
    project_perturbation(delta::Array, epsilon::Float64, norm::NormType)

Project perturbation to epsilon ball.
"""
function project_perturbation(delta::Array, epsilon::Float64, norm::NormType)::Array
    if norm == Linf
        return clamp.(delta, -epsilon, epsilon)
    elseif norm == L2
        norm_val = compute_norm(delta, L2)
        if norm_val > epsilon
            return delta .* (epsilon / norm_val)
        end
        return delta
    elseif norm == L1
        # Simplex projection for L1
        return project_l1_ball(delta, epsilon)
    else
        return delta
    end
end

"""
    project_l1_ball(v::Array, radius::Float64)

Project onto L1 ball using sorting-based algorithm.
"""
function project_l1_ball(v::Array, radius::Float64)::Array
    shape = size(v)
    v_flat = vec(abs.(v))
    
    if sum(v_flat) <= radius
        return v
    end
    
    # Sort in descending order
    sorted = sort(v_flat, rev=true)
    
    # Find threshold
    cumsum_sorted = cumsum(sorted)
    indices = 1:length(sorted)
    threshold_candidates = (cumsum_sorted .- radius) ./ indices
    
    # Find largest j where sorted[j] > threshold
    valid = sorted .> threshold_candidates
    j = findlast(valid)
    
    if j === nothing
        return zeros(shape)
    end
    
    threshold = threshold_candidates[j]
    
    # Apply soft thresholding
    result = sign.(v) .* max.(abs.(v) .- threshold, 0)
    return reshape(result, shape)
end

# ───────────────────────────────────────────────────────────────────────────────
#                              CLIPPING
# ───────────────────────────────────────────────────────────────────────────────

"""
    clip_perturbation(x_adv::Array, x_orig::Array, epsilon::Float64, 
                      norm::NormType, clip_min::Float64, clip_max::Float64)

Clip adversarial example to valid range and epsilon ball.
"""
function clip_perturbation(x_adv::Array, x_orig::Array, epsilon::Float64,
                           norm::NormType, clip_min::Float64 = 0.0, 
                           clip_max::Float64 = 1.0)::Array
    
    # Project perturbation
    delta = x_adv .- x_orig
    delta_projected = project_perturbation(delta, epsilon, norm)
    
    # Add back to original and clip to valid range
    x_clipped = x_orig .+ delta_projected
    x_clipped = clamp.(x_clipped, clip_min, clip_max)
    
    return x_clipped
end

# ───────────────────────────────────────────────────────────────────────────────
#                              NUMERICAL GRADIENTS
# ───────────────────────────────────────────────────────────────────────────────

"""
    numerical_gradient(f, x::Array; h::Float64 = 1e-4)

Compute numerical gradient using finite differences.
"""
function numerical_gradient(f, x::Array; h::Float64 = 1e-4)::Array
    grad = zeros(size(x))
    x_flat = vec(x)
    
    for i in eachindex(x_flat)
        x_plus = copy(x_flat)
        x_minus = copy(x_flat)
        x_plus[i] += h
        x_minus[i] -= h
        
        grad_flat = vec(grad)
        grad_flat[i] = (f(reshape(x_plus, size(x))) - f(reshape(x_minus, size(x)))) / (2h)
    end
    
    return grad
end

"""
    estimate_gradient_nes(f, x::Array, sigma::Float64, n_samples::Int)

Estimate gradient using Natural Evolution Strategies.
"""
function estimate_gradient_nes(f, x::Array, sigma::Float64, n_samples::Int)::Array
    grad = zeros(size(x))
    
    for _ in 1:n_samples
        noise = randn(size(x))
        
        f_plus = f(x .+ sigma .* noise)
        f_minus = f(x .- sigma .* noise)
        
        grad .+= (f_plus - f_minus) .* noise
    end
    
    grad ./= (2 * sigma * n_samples)
    return grad
end

"""
    estimate_gradient_spsa(f, x::Array, delta::Float64)

Estimate gradient using SPSA (Simultaneous Perturbation).
"""
function estimate_gradient_spsa(f, x::Array, delta::Float64)::Array
    # Rademacher random direction
    direction = 2.0 .* (rand(size(x)) .> 0.5) .- 1.0
    
    f_plus = f(x .+ delta .* direction)
    f_minus = f(x .- delta .* direction)
    
    grad = (f_plus - f_minus) ./ (2 * delta) .* direction
    return grad
end

# ───────────────────────────────────────────────────────────────────────────────
#                              LOSS FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

"""
    cross_entropy_loss(logits::Vector, target::Int)

Compute cross-entropy loss.
"""
function cross_entropy_loss(logits::Vector, target::Int)::Float64
    probs = softmax(logits)
    return -log(max(probs[target], 1e-10))
end

"""
    cw_loss(logits::Vector, target::Int, targeted::Bool, confidence::Float64)

Compute Carlini-Wagner loss.
"""
function cw_loss(logits::Vector, target::Int, targeted::Bool, confidence::Float64)::Float64
    target_logit = logits[target]
    
    # Get max logit excluding target
    other_logits = copy(logits)
    other_logits[target] = -Inf
    max_other = maximum(other_logits)
    
    if targeted
        # Minimize: max(max_other - target_logit + confidence, 0)
        return max(max_other - target_logit + confidence, 0.0)
    else
        # Minimize: max(target_logit - max_other + confidence, 0)
        return max(target_logit - max_other + confidence, 0.0)
    end
end

"""
    dlr_loss(logits::Vector, target::Int)

Compute Difference of Logits Ratio loss.
"""
function dlr_loss(logits::Vector, target::Int)::Float64
    sorted_logits = sort(logits, rev=true)
    
    z_target = logits[target]
    z_max = sorted_logits[1]
    z_second = sorted_logits[2]
    z_third = sorted_logits[3]
    
    if z_target == z_max
        return -(z_max - z_second) / (z_max - z_third + 1e-10)
    else
        return -(z_max - z_target) / (z_max - z_third + 1e-10)
    end
end

# ───────────────────────────────────────────────────────────────────────────────
#                              ACTIVATION FUNCTIONS
# ───────────────────────────────────────════════════════════════════════════════

"""
    softmax(x::Vector)

Compute softmax.
"""
function softmax(x::Vector)::Vector{Float64}
    x_max = maximum(x)
    exp_x = exp.(x .- x_max)
    return exp_x ./ sum(exp_x)
end

"""
    sigmoid(x)

Compute sigmoid.
"""
sigmoid(x) = 1.0 / (1.0 + exp(-x))

"""
    relu(x)

Compute ReLU.
"""
relu(x) = max(0, x)

# ───────────────────────────────────────────────────────────────────────────────
#                              RANDOM SAMPLING
# ───────────────────────────────────────────────────────────────────────────────

"""
    random_uniform_ball(shape, epsilon::Float64, norm::NormType)

Sample uniformly from epsilon ball.
"""
function random_uniform_ball(shape, epsilon::Float64, norm::NormType)::Array{Float32}
    if norm == Linf
        return Float32.(rand(shape...) .* 2epsilon .- epsilon)
    elseif norm == L2
        # Sample from unit sphere, scale by random radius
        direction = randn(Float32, shape...)
        direction ./= compute_norm(direction, L2)
        radius = rand()^(1/length(direction)) * epsilon
        return direction .* radius
    else
        return zeros(Float32, shape...)
    end
end

"""
    random_perturbation(shape, epsilon::Float64, norm::NormType)

Generate random perturbation.
"""
function random_perturbation(shape, epsilon::Float64, norm::NormType)::Array{Float32}
    delta = randn(Float32, shape...)
    return project_perturbation(delta, epsilon, norm)
end

# ───────────────────────────────────────────────────────────────────────────────
#                              METRICS
# ───────────────────────────────────────────────────────────────────────────────

"""
    attack_success_rate(predictions::Vector, original_labels::Vector, adversarial_labels::Vector)

Compute attack success rate.
"""
function attack_success_rate(original_labels::Vector, adversarial_labels::Vector)::Float64
    changed = count(o != a for (o, a) in zip(original_labels, adversarial_labels))
    return changed / length(original_labels)
end

"""
    perturbation_statistics(originals::Vector, adversarials::Vector)

Compute perturbation statistics.
"""
function perturbation_statistics(originals::Vector, adversarials::Vector)
    l2_norms = Float64[]
    linf_norms = Float64[]
    
    for (orig, adv) in zip(originals, adversarials)
        delta = adv .- orig
        push!(l2_norms, compute_norm(delta, L2))
        push!(linf_norms, compute_norm(delta, Linf))
    end
    
    return (
        mean_l2 = mean(l2_norms),
        std_l2 = std(l2_norms),
        mean_linf = mean(linf_norms),
        std_linf = std(linf_norms),
    )
end

# ───────────────────────────────────────────────────────────────────────────────
#                              TIMING
# ───────────────────────────────────────────────────────────────────────────────

"""
    @timed_block name expr

Time a block and log if verbose.
"""
macro timed_block(name, expr)
    quote
        local start_time = time()
        local result = $(esc(expr))
        local elapsed = time() - start_time
        CONFIG[].verbose && println(c(:dim), "  ", $(esc(name)), " completed in ", @sprintf("%.2fs", elapsed), c(:reset))
        result
    end
end
