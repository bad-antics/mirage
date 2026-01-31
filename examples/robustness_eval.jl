# ═══════════════════════════════════════════════════════════════════════════════
#                         MIRAGE - Robustness Evaluation Example
# ═══════════════════════════════════════════════════════════════════════════════
# Comprehensive robustness testing and defense evaluation
# ═══════════════════════════════════════════════════════════════════════════════

using Mirage

init()
banner()

println("\n" * "=" ^ 60)
println("          ROBUSTNESS EVALUATION DEMO")
println("=" ^ 60)

# ─────────────────────────────────────────────────────────────
# 1. Set up model and test data
# ─────────────────────────────────────────────────────────────

println("\n[1] Setting up model and test data...")

# Create model
model = create_surrogate(:resnet18, (32, 32, 3), 10)
println("    Model: ResNet-18")
println("    Input: $(model.input_shape)")
println("    Classes: $(model.num_classes)")

# Generate test data
n_samples = 50
test_data = [(rand(Float32, 32, 32, 3), rand(1:10)) for _ in 1:n_samples]
println("    Test samples: $n_samples")

# ─────────────────────────────────────────────────────────────
# 2. Evaluate clean accuracy
# ─────────────────────────────────────────────────────────────

println("\n[2] Evaluating clean accuracy...")

clean_acc = evaluate_clean_accuracy(model, test_data)
println("    Clean accuracy: $(round(clean_acc * 100, digits=1))%")

# ─────────────────────────────────────────────────────────────
# 3. Full robustness evaluation
# ─────────────────────────────────────────────────────────────

println("\n[3] Running comprehensive robustness evaluation...")

evaluation = evaluate_robustness(model,
                                 test_data=test_data,
                                 attacks=[:fgsm, :pgd, :cw],
                                 epsilons=[0.01, 0.03, 0.1])

display_evaluation(evaluation)

# ─────────────────────────────────────────────────────────────
# 4. Robustness profile analysis
# ─────────────────────────────────────────────────────────────

println("\n[4] Computing robustness profile...")

x_sample = test_data[1][1]
profile = robustness_profile(model, x_sample, n_samples=200)

display_robustness_profile(profile)

# ─────────────────────────────────────────────────────────────
# 5. Decision boundary analysis
# ─────────────────────────────────────────────────────────────

println("\n[5] Analyzing decision boundary...")

# Find minimum distance to boundary
boundary_info = minimum_boundary_distance(model, x_sample, n_directions=50)

println("    Minimum boundary distance: $(round(boundary_info["min_distance"], digits=4))")
println("    Mean boundary distance: $(round(boundary_info["mean_distance"], digits=4))")

# Map boundary in 2D
boundary_map = map_decision_boundary(model, x_sample, resolution=30)
display_boundary_map(boundary_map)

# ─────────────────────────────────────────────────────────────
# 6. Defense evaluation
# ─────────────────────────────────────────────────────────────

println("\n[6] Evaluating detection defense...")

# Generate adversarial samples
adversarial_data = []
for (x, y) in test_data[1:min(20, n_samples)]
    result = pgd(model, x, y, 0.05, 20)
    if result.success
        push!(adversarial_data, (result.adversarial, y))
    end
end

println("    Generated $(length(adversarial_data)) adversarial samples")

# Create detector
detector = create_detector([:feature_squeezing, :entropy], 
                           weights=[0.6, 0.4])

# Evaluate defense
defense_metrics = evaluate_defense(detector, model,
                                   test_data=test_data[1:20],
                                   adversarial_data=adversarial_data)

display_defense_metrics(defense_metrics)

# ─────────────────────────────────────────────────────────────
# 7. Certified accuracy (randomized smoothing)
# ─────────────────────────────────────────────────────────────

println("\n[7] Computing certified accuracy...")

certified = certified_accuracy(model, test_data[1:10],
                               sigma=0.25, n_samples=100)

println("    Certified accuracy: $(round(certified["certified_accuracy"] * 100, digits=1))%")
println("    Mean certified radius: $(round(certified["mean_certified_radius"], digits=4))")
println("    Median certified radius: $(round(certified["median_certified_radius"], digits=4))")

# ─────────────────────────────────────────────────────────────
# 8. Generate report
# ─────────────────────────────────────────────────────────────

println("\n[8] Generating robustness report...")

report = robustness_report(model, test_data=test_data,
                           output_path=joinpath(@__DIR__, "robustness_report.txt"))

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("              EVALUATION SUMMARY")
println("=" ^ 60)
println("""
Clean Accuracy:       $(round(clean_acc * 100, digits=1))%
Worst-Case Accuracy:  $(round(evaluation.worst_case_accuracy * 100, digits=1))%
Min Boundary Dist:    $(round(boundary_info["min_distance"], digits=4))
Detection F1 Score:   $(round(defense_metrics["f1_score"], digits=3))
Certified Accuracy:   $(round(certified["certified_accuracy"] * 100, digits=1))%

Report saved to: robustness_report.txt
""")
println("=" ^ 60)
