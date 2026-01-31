# ═══════════════════════════════════════════════════════════════════════════════
#                         MIRAGE - Advanced Attacks Example
# ═══════════════════════════════════════════════════════════════════════════════
# Demonstrates advanced attack techniques and analysis
# ═══════════════════════════════════════════════════════════════════════════════

using Mirage

init()
banner()

println("\n" * "=" ^ 60)
println("           ADVANCED ATTACKS DEMO")
println("=" ^ 60)

# ─────────────────────────────────────────────────────────────
# 1. Setup
# ─────────────────────────────────────────────────────────────

println("\n[1] Setting up model and input...")

model = create_surrogate(:resnet18, (224, 224, 3), 1000)
x = rand(Float32, 224, 224, 3)

original_pred = predict(model, x)
println("    Original class: $(original_pred.label)")
println("    Confidence: $(round(original_pred.confidence * 100, digits=1))%")

# ─────────────────────────────────────────────────────────────
# 2. Carlini & Wagner Attack (L2)
# ─────────────────────────────────────────────────────────────

println("\n[2] Running Carlini & Wagner L2 attack...")

result_cw = carlini_wagner(model, x, original_pred.label,
                           max_iter=100, confidence=10.0, c=0.01)

if result_cw.success
    println("    ✓ C&W attack successful")
    println("    L2 perturbation: $(round(result_cw.perturbation_norm, digits=4))")
    display_result(result_cw)
end

# ─────────────────────────────────────────────────────────────
# 3. DeepFool Attack (minimal perturbation)
# ─────────────────────────────────────────────────────────────

println("\n[3] Running DeepFool attack...")

result_df = deepfool(model, x, max_iter=50)

if result_df.success
    println("    ✓ DeepFool found minimal perturbation")
    println("    L2 perturbation: $(round(result_df.perturbation_norm, digits=4))")
    display_result(result_df)
end

# ─────────────────────────────────────────────────────────────
# 4. Auto-PGD (adaptive step size)
# ─────────────────────────────────────────────────────────────

println("\n[4] Running Auto-PGD attack...")

result_apgd = apgd(model, x, original_pred.label, 0.03, 100)

if result_apgd.success
    println("    ✓ Auto-PGD attack successful")
    display_result(result_apgd)
end

# ─────────────────────────────────────────────────────────────
# 5. AutoAttack (ensemble)
# ─────────────────────────────────────────────────────────────

println("\n[5] Running AutoAttack ensemble...")

result_aa = auto_attack(model, x, original_pred.label, 0.03)

if result_aa.success
    println("    ✓ AutoAttack successful")
    display_result(result_aa)
end

# ─────────────────────────────────────────────────────────────
# 6. HopSkipJump (decision-based)
# ─────────────────────────────────────────────────────────────

println("\n[6] Running HopSkipJump attack...")

result_hsj = hopskipjump(model, x, max_iter=50, max_queries=1000)

if result_hsj.success
    println("    ✓ HopSkipJump attack successful")
    println("    Queries used: $(result_hsj.queries)")
    display_result(result_hsj)
end

# ─────────────────────────────────────────────────────────────
# 7. SimBA (Simple Black-box Attack)
# ─────────────────────────────────────────────────────────────

println("\n[7] Running SimBA attack...")

result_simba = simba(model, x, epsilon=0.1, max_queries=500)

if result_simba.success
    println("    ✓ SimBA attack successful")
    println("    Queries used: $(result_simba.queries)")
    display_result(result_simba)
end

# ─────────────────────────────────────────────────────────────
# 8. Saliency-guided attack
# ─────────────────────────────────────────────────────────────

println("\n[8] Saliency-guided perturbation...")

# Compute saliency
saliency = gradient_saliency(model, x)
display_saliency(saliency)

# Use saliency to guide perturbation
saliency_mask = saliency.saliency .> 0.5  # Focus on important regions
println("    Salient pixels: $(sum(saliency_mask)) of $(prod(size(x)[1:2]))")

# ─────────────────────────────────────────────────────────────
# 9. Integrated Gradients analysis
# ─────────────────────────────────────────────────────────────

println("\n[9] Computing Integrated Gradients...")

ig_saliency = integrated_gradients(model, x, steps=30)
display_saliency(ig_saliency)

# ─────────────────────────────────────────────────────────────
# 10. Compare all attack results
# ─────────────────────────────────────────────────────────────

println("\n[10] Attack Comparison:")
println("=" ^ 70)
println("Attack          | Success | L2 Norm    | Queries | Time (s)")
println("-" ^ 70)

results = [
    ("C&W", result_cw),
    ("DeepFool", result_df),
    ("Auto-PGD", result_apgd),
    ("AutoAttack", result_aa),
    ("HopSkipJump", result_hsj),
    ("SimBA", result_simba)
]

for (name, result) in results
    success = result.success ? "✓" : "✗"
    l2 = round(result.perturbation_norm, digits=4)
    queries = result.queries
    time = round(result.time_elapsed, digits=2)
    
    println("$(rpad(name, 15)) | $(rpad(success, 7)) | $(rpad(l2, 10)) | $(rpad(queries, 7)) | $time")
end

println("=" ^ 70)

# ─────────────────────────────────────────────────────────────
# 11. Boundary walk visualization
# ─────────────────────────────────────────────────────────────

println("\n[11] Walking decision boundary...")

boundary_path = walk_boundary(model, x, steps=50, step_size=0.01)
println("    Traced $(length(boundary_path)) points along boundary")

# ─────────────────────────────────────────────────────────────
# 12. Adversarial subspace analysis
# ─────────────────────────────────────────────────────────────

println("\n[12] Finding adversarial subspace...")

subspace = find_adversarial_subspace(model, x, n_components=5)

if !isempty(subspace["components"])
    println("    Found $(length(subspace["components"])) principal adversarial directions")
    println("    Explained variance: $(round(subspace["explained_variance"] * 100, digits=1))%")
    
    for (i, var) in enumerate(subspace["variances"])
        println("    Component $i: $(round(var * 100, digits=1))%")
    end
end

println("\n" * "=" ^ 60)
println("            ADVANCED ATTACKS DEMO COMPLETE")
println("=" ^ 60)
