# ═══════════════════════════════════════════════════════════════════════════════
#                         MIRAGE - Model Extraction Example
# ═══════════════════════════════════════════════════════════════════════════════
# Demonstrates model stealing attacks
# ═══════════════════════════════════════════════════════════════════════════════

using Mirage

init()
banner()

println("\n" * "=" ^ 60)
println("           MODEL EXTRACTION ATTACK DEMO")
println("=" ^ 60)

# ─────────────────────────────────────────────────────────────
# 1. Set up target model (simulates API endpoint)
# ─────────────────────────────────────────────────────────────

println("\n[1] Setting up target model (black-box API)...")

# Simulate a remote API
target = connect_custom("http://api.example.com/predict",
                        input_shape=(32, 32, 3),
                        num_classes=10,
                        rate_limit=100)

println("    Target: $(target.url)")
println("    Input: $(target.input_shape) → $(target.num_classes) classes")
println("    Rate limit: $(target.rate_limit) queries/sec")

# ─────────────────────────────────────────────────────────────
# 2. Knockoff Nets extraction
# ─────────────────────────────────────────────────────────────

println("\n[2] Running Knockoff Nets extraction...")
println("    Budget: 5000 queries")

surrogate_knockoff = knockoff_nets(target,
                                   architecture=:resnet18,
                                   budget=5000,
                                   batch_size=128,
                                   epochs=10)

model_summary(surrogate_knockoff)

# ─────────────────────────────────────────────────────────────
# 3. JBDA extraction (with seed data)
# ─────────────────────────────────────────────────────────────

println("\n[3] Running JBDA extraction...")

# Create some seed data
seed_samples = [rand(Float32, 32, 32, 3) for _ in 1:100]
seed_labels = rand(1:10, 100)
seed_data = Dataset(seed_samples, seed_labels, :seed)

surrogate_jbda = jbda_extract(target,
                              seed_data=seed_data,
                              augmentation_factor=5,
                              budget=2000)

model_summary(surrogate_jbda)

# ─────────────────────────────────────────────────────────────
# 4. ActiveThief extraction
# ─────────────────────────────────────────────────────────────

println("\n[4] Running ActiveThief extraction...")

surrogate_active = active_thief(target,
                                strategy=:entropy,
                                budget=3000,
                                batch_size=50)

model_summary(surrogate_active)

# ─────────────────────────────────────────────────────────────
# 5. Compare extraction methods
# ─────────────────────────────────────────────────────────────

println("\n[5] Extraction Results Comparison:")
println("=" ^ 60)
println("Method           | Fidelity | Queries | Train Samples")
println("-" ^ 60)
println("Knockoff Nets    | $(rpad(round(surrogate_knockoff.fidelity*100, digits=1), 8))% | $(rpad(surrogate_knockoff.queries_used, 7)) | $(surrogate_knockoff.training_samples)")
println("JBDA             | $(rpad(round(surrogate_jbda.fidelity*100, digits=1), 8))% | $(rpad(surrogate_jbda.queries_used, 7)) | $(surrogate_jbda.training_samples)")
println("ActiveThief      | $(rpad(round(surrogate_active.fidelity*100, digits=1), 8))% | $(rpad(surrogate_active.queries_used, 7)) | $(surrogate_active.training_samples)")
println("=" ^ 60)

# ─────────────────────────────────────────────────────────────
# 6. Use surrogate for transfer attacks
# ─────────────────────────────────────────────────────────────

println("\n[6] Using surrogate for transfer attack...")

# Use the best surrogate
best_surrogate = surrogate_knockoff.fidelity > surrogate_jbda.fidelity ? surrogate_knockoff : surrogate_jbda

# Generate adversarial on surrogate
x_test = rand(Float32, 32, 32, 3)
pred_target = predict(target, x_test)

println("    Original target prediction: Class $(pred_target.label)")

# White-box attack on surrogate
result = pgd(best_surrogate, x_test, pred_target.label, 0.03, 40)

if result.success
    # Test transfer to target
    x_adv = result.adversarial
    pred_adv = predict(target, x_adv)
    
    println("    Adversarial prediction on target: Class $(pred_adv.label)")
    
    if pred_adv.label != pred_target.label
        println("    ✓ Transfer attack successful!")
    else
        println("    ✗ Transfer attack failed (surrogate mismatch)")
    end
end

# ─────────────────────────────────────────────────────────────
# 7. Save surrogate for later use
# ─────────────────────────────────────────────────────────────

println("\n[7] Saving surrogate model...")

save_path = joinpath(@__DIR__, "extracted_model.toml")
save_surrogate(best_surrogate, save_path)

println("\n" * "=" ^ 60)
println("            EXTRACTION DEMO COMPLETE")
println("=" ^ 60)
