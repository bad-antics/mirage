# ═══════════════════════════════════════════════════════════════════════════════
#                                    MIRAGE
#                        Adversarial Machine Learning Toolkit
# ═══════════════════════════════════════════════════════════════════════════════
# Model Extraction • Adversarial Examples • Neural Network Probing
# ═══════════════════════════════════════════════════════════════════════════════

module Mirage

using Dates
using HTTP
using JSON3
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML
using UUIDs

# ───────────────────────────────────────────────────────────────────────────────
#                              EXPORTS
# ───────────────────────────────────────────────────────────────────────────────

# Core
export MirageConfig, AttackConfig, AttackResult
export init, configure, status

# Models
export Model, RemoteModel, LocalModel, SurrogateModel
export load_model, predict, gradient

# White-box attacks
export fgsm, pgd, carlini_wagner, deepfool, auto_attack
export apgd, fab_attack

# Black-box attacks
export square_attack, hopskipjump, boundary_attack
export simba, qeba, nes_attack, spsa

# Model extraction
export extract_model, knockoff_nets, jbda_extract, active_thief
export evaluate_fidelity

# Analysis
export gradient_saliency, integrated_gradients, lime
export map_decision_boundary, probe_neurons
export most_activated, layer_relevance

# Defense testing
export evaluate_robustness, robustness_report
export test_defense, bypass_preprocessing

# Metrics
export attack_success_rate, perturbation_norm
export evaluate_attack, transferability

# Display
export banner, display_result, display_report

# ───────────────────────────────────────────────────────────────────────────────
#                              CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────

const VERSION = v"1.0.0"
const AUTHOR = "bad-antics"

# Terminal colors
const COLORS = Dict(
    :reset => "\033[0m",
    :bold => "\033[1m",
    :dim => "\033[2m",
    :red => "\033[31m",
    :green => "\033[32m",
    :yellow => "\033[33m",
    :blue => "\033[34m",
    :magenta => "\033[35m",
    :cyan => "\033[36m",
    :white => "\033[37m",
    :bg_red => "\033[41m",
    :bg_green => "\033[42m",
    :bg_blue => "\033[44m",
    :bg_magenta => "\033[45m",
)

# ───────────────────────────────────────────────────────────────────────────────
#                              TYPES
# ───────────────────────────────────────────────────────────────────────────────

"""
Norm types for perturbation constraints.
"""
@enum NormType begin
    L0
    L1
    L2
    Linf
end

"""
Attack status.
"""
@enum AttackStatus begin
    SUCCESS
    FAILED
    PARTIAL
    TIMEOUT
end

"""
Configuration for Mirage.
"""
mutable struct MirageConfig
    device::Symbol
    threads::Int
    precision::Type
    verbose::Bool
    log_queries::Bool
    cache_gradients::Bool
    nullsec_integration::Bool
end

function MirageConfig()
    MirageConfig(
        :cpu,
        Threads.nthreads(),
        Float32,
        true,
        true,
        true,
        false
    )
end

"""
Attack configuration.
"""
struct AttackConfig
    norm::NormType
    epsilon::Float64
    targeted::Bool
    target_class::Union{Int, Nothing}
    confidence::Float64
    max_iterations::Int
    early_stop::Bool
end

function AttackConfig(;
    norm::NormType = Linf,
    epsilon::Float64 = 0.03,
    targeted::Bool = false,
    target_class::Union{Int, Nothing} = nothing,
    confidence::Float64 = 0.0,
    max_iterations::Int = 100,
    early_stop::Bool = true
)
    AttackConfig(norm, epsilon, targeted, target_class, confidence, max_iterations, early_stop)
end

"""
Result of an attack.
"""
struct AttackResult
    status::AttackStatus
    original::Vector{Float32}
    adversarial::Vector{Float32}
    original_label::Int
    adversarial_label::Int
    perturbation_l2::Float64
    perturbation_linf::Float64
    queries_used::Int
    iterations::Int
    confidence_original::Float64
    confidence_adversarial::Float64
    time_elapsed::Float64
end

"""
Robustness evaluation result.
"""
struct RobustnessResult
    attack_name::Symbol
    epsilon::Float64
    success_rate::Float64
    avg_perturbation::Float64
    avg_queries::Float64
    samples_tested::Int
end

# Global config
const CONFIG = Ref{MirageConfig}(MirageConfig())

# ───────────────────────────────────────────────────────────────────────────────
#                              INCLUDES
# ───────────────────────────────────────────────────────────────────────────────

include("core/Types.jl")
include("core/Config.jl")
include("core/Display.jl")
include("core/Utils.jl")

include("models/Loaders.jl")
include("models/Remote.jl")
include("models/Surrogate.jl")

include("attacks/WhiteBox.jl")
include("attacks/BlackBox.jl")
include("attacks/Extraction.jl")

include("analysis/Saliency.jl")
include("analysis/Probing.jl")
include("analysis/Boundary.jl")

include("defenses/Detection.jl")
include("defenses/Evaluation.jl")

# ───────────────────────────────────────────────────────────────────────────────
#                              INITIALIZATION
# ───────────────────────────────────────────────────────────────────────────────

"""
    init(;kwargs...)

Initialize Mirage with optional configuration.
"""
function init(;
    device::Symbol = :cpu,
    threads::Int = Threads.nthreads(),
    verbose::Bool = true
)::MirageConfig
    
    config = CONFIG[]
    config.device = device
    config.threads = threads
    config.verbose = verbose
    
    # Check for NullSec environment
    config.nullsec_integration = detect_nullsec()
    
    if verbose
        printstyled("✓ ", color = :green, bold = true)
        println("Mirage v$VERSION initialized")
        println("  Device: $device | Threads: $threads")
        
        if config.nullsec_integration
            printstyled("  NullSec: ", color = :cyan)
            println("Detected")
        end
    end
    
    return config
end

"""
    configure(;kwargs...)

Update Mirage configuration.
"""
function configure(;
    device::Union{Symbol, Nothing} = nothing,
    threads::Union{Int, Nothing} = nothing,
    precision::Union{Type, Nothing} = nothing,
    verbose::Union{Bool, Nothing} = nothing,
    log_queries::Union{Bool, Nothing} = nothing,
    cache_gradients::Union{Bool, Nothing} = nothing
)::MirageConfig
    
    config = CONFIG[]
    
    device !== nothing && (config.device = device)
    threads !== nothing && (config.threads = threads)
    precision !== nothing && (config.precision = precision)
    verbose !== nothing && (config.verbose = verbose)
    log_queries !== nothing && (config.log_queries = log_queries)
    cache_gradients !== nothing && (config.cache_gradients = cache_gradients)
    
    return config
end

"""
    status()

Display current Mirage status.
"""
function status()
    config = CONFIG[]
    
    println()
    println(c(:magenta), "╔═══════════════════════════════════════════════════════════════╗", c(:reset))
    println(c(:magenta), "║                      MIRAGE STATUS                            ║", c(:reset))
    println(c(:magenta), "╠═══════════════════════════════════════════════════════════════╣", c(:reset))
    println(c(:magenta), "║", c(:reset), " Version:     ", c(:cyan), "v$VERSION", c(:reset))
    println(c(:magenta), "║", c(:reset), " Device:      ", c(:yellow), "$(config.device)", c(:reset))
    println(c(:magenta), "║", c(:reset), " Threads:     ", c(:white), "$(config.threads)", c(:reset))
    println(c(:magenta), "║", c(:reset), " Precision:   ", c(:white), "$(config.precision)", c(:reset))
    println(c(:magenta), "║", c(:reset), " Verbose:     ", config.verbose ? "$(c(:green))Yes" : "$(c(:dim))No", c(:reset))
    println(c(:magenta), "║", c(:reset), " NullSec:     ", config.nullsec_integration ? "$(c(:green))Connected" : "$(c(:dim))Not detected", c(:reset))
    println(c(:magenta), "╚═══════════════════════════════════════════════════════════════╝", c(:reset))
end

# ───────────────────────────────────────────────────────────────────────────────
#                              BANNER
# ───────────────────────────────────────────────────────────────────────────────

const BANNER = """

    $(c(:magenta))███╗   ███╗██╗██████╗  █████╗  ██████╗ ███████╗$(c(:reset))
    $(c(:magenta))████╗ ████║██║██╔══██╗██╔══██╗██╔════╝ ██╔════╝$(c(:reset))
    $(c(:magenta))██╔████╔██║██║██████╔╝███████║██║  ███╗█████╗  $(c(:reset))
    $(c(:magenta))██║╚██╔╝██║██║██╔══██╗██╔══██║██║   ██║██╔══╝  $(c(:reset))
    $(c(:magenta))██║ ╚═╝ ██║██║██║  ██║██║  ██║╚██████╔╝███████╗$(c(:reset))
    $(c(:magenta))╚═╝     ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝$(c(:reset))
                                                    
    $(c(:dim))[ Adversarial ML Toolkit | v$VERSION | bad-antics ]$(c(:reset))

"""

"""
    banner()

Display the Mirage banner.
"""
function banner()
    print(BANNER)
end

# ───────────────────────────────────────────────────────────────────────────────
#                              HELPERS
# ───────────────────────────────────────────────────────────────────────────────

"""Color shorthand."""
c(name::Symbol) = get(COLORS, name, COLORS[:reset])

"""Detect NullSec environment."""
function detect_nullsec()::Bool
    isdir(expanduser("~/nullsec")) && return true
    haskey(ENV, "NULLSEC_HOME") && return true
    return false
end

"""Check if NullSec is available."""
nullsec_available() = CONFIG[].nullsec_integration

end # module
