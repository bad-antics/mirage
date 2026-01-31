# ═══════════════════════════════════════════════════════════════════════════════
#                         MIRAGE - Quick Start Example
# ═══════════════════════════════════════════════════════════════════════════════
# Demonstrates basic adversarial attack generation
# ═══════════════════════════════════════════════════════════════════════════════

using Mirage

# Initialize Mirage
init()
banner()

println("\n" * "=" ^ 60)
println("           MIRAGE QUICK START EXAMPLE")
println("=" ^ 60)

# ─────────────────────────────────────────────────────────────
# 1. Create a mock model for demonstration
# ─────────────────────────────────────────────────────────────

println("\n[1] Setting up mock model...")

# Create a simple mock remote model
model = create_mock_model((28, 28, 1), 10)
println("    Mock model created: MNIST-like (28×28×1) → 10 classes")

# ─────────────────────────────────────────────────────────────
# 2. Generate a sample input
# ─────────────────────────────────────────────────────────────

println("\n[2] Creating sample input...")

# Random "image"
x = rand(Float32, 28, 28, 1)
println("    Input shape: $(size(x))")

# Get original prediction
original_pred = predict(model, x)
println("    Original prediction: Class $(original_pred.label)")
println("    Confidence: $(round(original_pred.confidence * 100, digits=1))%")

# ─────────────────────────────────────────────────────────────
# 3. Generate adversarial example with FGSM
# ─────────────────────────────────────────────────────────────

println("\n[3] Running FGSM attack...")

# Configure attack
epsilon = 0.1
config = AttackConfig(epsilon=epsilon, max_iter=1)

# Run FGSM
result = fgsm(model, x, original_pred.label, epsilon)

if result.success
    println("    ✓ Attack successful!")
    println("    Perturbation L2 norm: $(round(result.perturbation_norm, digits=4))")
    println("    Queries used: $(result.queries)")
    
    # Check new prediction
    x_adv = result.adversarial
    new_pred = predict(model, x_adv)
    println("    New prediction: Class $(new_pred.label)")
else
    println("    ✗ Attack failed")
end

# ─────────────────────────────────────────────────────────────
# 4. Generate adversarial example with PGD
# ─────────────────────────────────────────────────────────────

println("\n[4] Running PGD attack...")

# PGD parameters
result_pgd = pgd(model, x, original_pred.label, epsilon, 40)

if result_pgd.success
    println("    ✓ PGD attack successful!")
    println("    Perturbation L2 norm: $(round(result_pgd.perturbation_norm, digits=4))")
    println("    Iterations: $(result_pgd.queries)")
    display_result(result_pgd)
else
    println("    ✗ PGD attack failed")
end

# ─────────────────────────────────────────────────────────────
# 5. Black-box attack (Square Attack)
# ─────────────────────────────────────────────────────────────

println("\n[5] Running Square Attack (black-box)...")

result_square = square_attack(model, x, original_pred.label, 
                              epsilon=epsilon, max_queries=500)

if result_square.success
    println("    ✓ Square attack successful!")
    println("    Queries used: $(result_square.queries)")
    display_result(result_square)
else
    println("    ✗ Square attack failed (may need more queries)")
end

# ─────────────────────────────────────────────────────────────
# 6. Saliency analysis
# ─────────────────────────────────────────────────────────────

println("\n[6] Computing gradient saliency...")

# Create surrogate for gradient computation
surrogate = create_surrogate(:simple_cnn, (28, 28, 1), 10)

saliency = gradient_saliency(surrogate, x)
println("    Saliency computed for class $(saliency.target_class)")
println("    Max saliency value: $(round(saliency.max_value, digits=4))")

# ─────────────────────────────────────────────────────────────
# 7. Detection test
# ─────────────────────────────────────────────────────────────

println("\n[7] Running adversarial detection...")

# Test on clean input
detection_clean = detect_adversarial(surrogate, x, 
                                     methods=[:feature_squeezing, :entropy])
println("    Clean input:")
println("    - Is adversarial: $(detection_clean["is_adversarial"])")
println("    - Score: $(round(detection_clean["aggregate_score"], digits=3))")

# Test on adversarial (if we have one)
if result_pgd.success
    detection_adv = detect_adversarial(surrogate, result_pgd.adversarial,
                                       methods=[:feature_squeezing, :entropy])
    println("    Adversarial input:")
    println("    - Is adversarial: $(detection_adv["is_adversarial"])")
    println("    - Score: $(round(detection_adv["aggregate_score"], digits=3))")
end

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("                    EXAMPLE COMPLETE")
println("=" ^ 60)
println("""

Mirage provides:
  • White-box attacks: FGSM, PGD, C&W, DeepFool, Auto-PGD
  • Black-box attacks: Square, HopSkipJump, Boundary, SimBA
  • Model extraction: Knockoff Nets, JBDA, ActiveThief
  • Analysis: Saliency maps, neural probing, boundary analysis
  • Defenses: Detection methods, robustness evaluation

For more examples, see:
  • examples/model_extraction.jl
  • examples/robustness_eval.jl
  • examples/advanced_attacks.jl

""")
