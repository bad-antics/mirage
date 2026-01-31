# ═══════════════════════════════════════════════════════════════════════════════
#                              MIRAGE - Test Suite
# ═══════════════════════════════════════════════════════════════════════════════

using Test

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Mirage

@testset "Mirage Tests" begin
    
    @testset "Core Types" begin
        # Test NormType enum
        @test L0 isa NormType
        @test L1 isa NormType
        @test L2 isa NormType
        @test Linf isa NormType
        
        # Test AttackStatus enum
        @test SUCCESS isa AttackStatus
        @test FAILED isa AttackStatus
        @test PARTIAL isa AttackStatus
        @test TIMEOUT isa AttackStatus
        
        # Test MirageConfig
        config = MirageConfig()
        @test config.verbose == true
        @test config.default_epsilon == 0.03
        
        # Test AttackConfig
        attack_config = AttackConfig()
        @test attack_config.epsilon == 0.03
        @test attack_config.max_iter == 100
    end
    
    @testset "Utility Functions" begin
        # Test norm computation
        x = [1.0, 2.0, 3.0]
        
        @test compute_norm(x, L0) == 3  # Non-zero elements
        @test compute_norm(x, L1) ≈ 6.0
        @test compute_norm(x, L2) ≈ sqrt(14)
        @test compute_norm(x, Linf) ≈ 3.0
        
        # Test projection
        delta = [0.5, 0.5, 0.5]
        projected = project_perturbation(delta, L2, 0.5)
        @test compute_norm(projected, L2) <= 0.5 + 1e-6
        
        # Test softmax
        logits = [1.0, 2.0, 3.0]
        probs = softmax(logits)
        @test sum(probs) ≈ 1.0
        @test all(probs .>= 0)
        @test argmax(probs) == 3
        
        # Test sigmoid
        @test sigmoid(0.0) ≈ 0.5
        @test sigmoid(100.0) ≈ 1.0 atol=1e-6
        @test sigmoid(-100.0) ≈ 0.0 atol=1e-6
        
        # Test ReLU
        @test relu(5.0) == 5.0
        @test relu(-5.0) == 0.0
    end
    
    @testset "Display Functions" begin
        # Test progress bar
        bar = progress_bar(50, 100, 20)
        @test length(bar) == 20
        @test occursin("█", bar)
        
        # Test spinner
        for i in 1:4
            s = spinner(i)
            @test length(s) == 1
        end
    end
    
    @testset "Attack Configuration" begin
        # Test preset loading
        fast_config = ATTACK_PRESETS[:fast]
        @test fast_config[:max_iter] < ATTACK_PRESETS[:strong][:max_iter]
        
        strong_config = ATTACK_PRESETS[:strong]
        @test strong_config[:epsilon] > ATTACK_PRESETS[:fast][:epsilon]
        
        # Test epsilon guidelines
        @test haskey(EPSILON_GUIDELINES, :imagenet)
        @test haskey(EPSILON_GUIDELINES, :cifar10)
        @test haskey(EPSILON_GUIDELINES, :mnist)
    end
    
    @testset "Attack Registry" begin
        # Check registered attacks
        @test haskey(ATTACK_REGISTRY, :fgsm)
        @test haskey(ATTACK_REGISTRY, :pgd)
        @test haskey(ATTACK_REGISTRY, :cw)
        @test haskey(ATTACK_REGISTRY, :deepfool)
        @test haskey(ATTACK_REGISTRY, :square)
        @test haskey(ATTACK_REGISTRY, :hopskipjump)
        
        # Check attack info
        fgsm_info = ATTACK_REGISTRY[:fgsm]
        @test fgsm_info.name == "Fast Gradient Sign Method"
        @test fgsm_info.category == :whitebox
    end
    
    @testset "Model Types" begin
        # Test LocalModel structure
        @test LocalModel <: Any
        
        # Test SurrogateModel structure  
        @test SurrogateModel <: Any
        
        # Test Prediction structure
        pred = Prediction(1, 0.95, [0.95, 0.05])
        @test pred.label == 1
        @test pred.confidence ≈ 0.95
        @test sum(pred.probabilities) ≈ 1.0
    end
    
    @testset "Loss Functions" begin
        probs = [0.1, 0.2, 0.7]
        
        # Cross entropy loss
        ce_loss = cross_entropy_loss(probs, 3)
        @test ce_loss >= 0
        
        # CW loss  
        cw = cw_loss(probs, 3)
        @test cw isa Float64
        
        # DLR loss
        dlr = dlr_loss(probs, 3)
        @test dlr isa Float64
    end
    
    @testset "Metrics" begin
        # Test attack success rate
        results = [
            AttackResult(true, rand(3), 0.01, 10, 0.1),
            AttackResult(false, rand(3), 0.0, 10, 0.1),
            AttackResult(true, rand(3), 0.02, 10, 0.1)
        ]
        
        asr = attack_success_rate(results)
        @test asr ≈ 2/3
        
        # Test perturbation statistics
        perturbations = [[0.1, 0.2], [0.3, 0.4], [0.2, 0.3]]
        stats = perturbation_statistics(perturbations)
        
        @test haskey(stats, "mean_l2")
        @test haskey(stats, "max_l2")
        @test haskey(stats, "mean_linf")
    end
    
    @testset "Initialization" begin
        # Test init function
        @test init() == true
        
        # Test status
        status_info = status()
        @test haskey(status_info, "version")
        @test haskey(status_info, "initialized")
    end
    
end

println("\n✓ All tests passed!")
