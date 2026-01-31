# ═══════════════════════════════════════════════════════════════════════════════
#                              MIRAGE - Core Types
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
#                              MODEL TYPES
# ───────────────────────────────────────────────────────────────────────────────

"""
Abstract base type for all models.
"""
abstract type Model end

"""
Local model loaded from file.
"""
struct LocalModel <: Model
    path::String
    framework::Symbol  # :onnx, :pytorch, :tensorflow, :flux
    input_shape::Tuple
    num_classes::Int
    weights::Dict{String, Array}
    layers::Vector{Symbol}
end

"""
Remote model accessed via API.
"""
mutable struct RemoteModel <: Model
    endpoint::String
    headers::Dict{String, String}
    input_shape::Tuple
    num_classes::Int
    query_count::Int
    query_limit::Union{Int, Nothing}
    last_response::Union{Dict, Nothing}
end

function RemoteModel(endpoint::String;
    headers::Dict{String, String} = Dict{String, String}(),
    input_shape::Tuple = (224, 224, 3),
    num_classes::Int = 1000,
    query_limit::Union{Int, Nothing} = nothing
)
    RemoteModel(endpoint, headers, input_shape, num_classes, 0, query_limit, nothing)
end

"""
Surrogate model trained through extraction.
"""
mutable struct SurrogateModel <: Model
    architecture::Symbol
    input_shape::Tuple
    num_classes::Int
    weights::Dict{String, Array}
    fidelity::Float64
    queries_used::Int
    training_samples::Int
end

# ───────────────────────────────────────────────────────────────────────────────
#                              INPUT/OUTPUT TYPES
# ───────────────────────────────────────────────────────────────────────────────

"""
Input sample with optional label.
"""
struct Sample
    data::Array{Float32}
    label::Union{Int, Nothing}
    id::String
end

Sample(data::Array) = Sample(Float32.(data), nothing, string(uuid4()))

"""
Model prediction result.
"""
struct Prediction
    label::Int
    confidence::Float64
    probabilities::Vector{Float64}
    logits::Union{Vector{Float64}, Nothing}
end

"""
Gradient information.
"""
struct GradientInfo
    input_gradient::Array{Float32}
    loss_value::Float64
    layer_gradients::Dict{String, Array{Float32}}
end

# ───────────────────────────────────────────────────────────────────────────────
#                              ATTACK TYPES
# ───────────────────────────────────────────────────────────────────────────────

"""
Attack method specification.
"""
struct AttackMethod
    name::Symbol
    category::Symbol  # :whitebox, :blackbox, :extraction
    requires_gradient::Bool
    requires_logits::Bool
    requires_labels::Bool
    description::String
end

const ATTACK_REGISTRY = Dict{Symbol, AttackMethod}(
    # White-box
    :fgsm => AttackMethod(:fgsm, :whitebox, true, false, true, "Fast Gradient Sign Method"),
    :pgd => AttackMethod(:pgd, :whitebox, true, false, true, "Projected Gradient Descent"),
    :cw => AttackMethod(:cw, :whitebox, true, true, true, "Carlini-Wagner L2/Linf"),
    :deepfool => AttackMethod(:deepfool, :whitebox, true, true, false, "Minimal perturbation finder"),
    :autoattack => AttackMethod(:autoattack, :whitebox, true, true, true, "Ensemble of strong attacks"),
    :apgd => AttackMethod(:apgd, :whitebox, true, true, true, "Auto-PGD"),
    :fab => AttackMethod(:fab, :whitebox, true, true, true, "Fast Adaptive Boundary"),
    
    # Black-box
    :square => AttackMethod(:square, :blackbox, false, false, true, "Query-efficient score-based"),
    :hopskipjump => AttackMethod(:hopskipjump, :blackbox, false, false, false, "Decision-based attack"),
    :boundary => AttackMethod(:boundary, :blackbox, false, false, false, "Decision boundary attack"),
    :simba => AttackMethod(:simba, :blackbox, false, false, true, "Simple Black-box Attack"),
    :qeba => AttackMethod(:qeba, :blackbox, false, false, false, "Query-Efficient Boundary Attack"),
    :nes => AttackMethod(:nes, :blackbox, false, false, true, "Natural Evolution Strategies"),
    :spsa => AttackMethod(:spsa, :blackbox, false, false, true, "Simultaneous Perturbation"),
    
    # Extraction
    :knockoff => AttackMethod(:knockoff, :extraction, false, false, false, "Knockoff Nets"),
    :jbda => AttackMethod(:jbda, :extraction, false, false, false, "Jacobian-Based Data Augmentation"),
    :activethief => AttackMethod(:activethief, :extraction, false, false, false, "Active learning extraction"),
)

"""
Perturbation constraint.
"""
struct PerturbationConstraint
    norm::NormType
    epsilon::Float64
    clip_min::Float64
    clip_max::Float64
end

PerturbationConstraint(;
    norm::NormType = Linf,
    epsilon::Float64 = 0.03,
    clip_min::Float64 = 0.0,
    clip_max::Float64 = 1.0
) = PerturbationConstraint(norm, epsilon, clip_min, clip_max)

# ───────────────────────────────────────────────────────────────────────────────
#                              ANALYSIS TYPES
# ───────────────────────────────────────────────────────────────────────────────

"""
Saliency map result.
"""
struct SaliencyMap
    attribution::Array{Float32}
    method::Symbol
    target_class::Int
    normalized::Bool
end

"""
Decision boundary point.
"""
struct BoundaryPoint
    position::Vector{Float64}
    class_a::Int
    class_b::Int
    distance::Float64
end

"""
Neuron activation info.
"""
struct NeuronActivation
    layer::String
    index::Int
    activation::Float64
    input_receptive_field::Union{Array, Nothing}
end

"""
Layer probing result.
"""
struct LayerProbe
    layer_name::String
    activations::Array{Float32}
    mean_activation::Float64
    sparsity::Float64
    top_neurons::Vector{Int}
end

# ───────────────────────────────────────────────────────────────────────────────
#                              DEFENSE TYPES
# ───────────────────────────────────────────────────────────────────────────────

"""
Defense mechanism.
"""
struct Defense
    name::Symbol
    category::Symbol  # :preprocessing, :training, :detection, :certified
    description::String
end

"""
Defense evaluation result.
"""
struct DefenseEvaluation
    defense::Defense
    attack::AttackMethod
    clean_accuracy::Float64
    robust_accuracy::Float64
    detection_rate::Float64
    false_positive_rate::Float64
end

# ───────────────────────────────────────────────────────────────────────────────
#                              METRICS
# ───────────────────────────────────────────────────────────────────────────────

"""
Attack metrics.
"""
struct AttackMetrics
    success_rate::Float64
    avg_perturbation_l2::Float64
    avg_perturbation_linf::Float64
    avg_queries::Float64
    avg_iterations::Float64
    avg_time::Float64
    confidence_drop::Float64
end

"""
Extraction metrics.
"""
struct ExtractionMetrics
    fidelity::Float64           # Agreement with target
    accuracy::Float64           # On test set
    queries_used::Int
    training_time::Float64
    model_size::Int             # Parameters
end

# ───────────────────────────────────────────────────────────────────────────────
#                              DATASET TYPES
# ───────────────────────────────────────────────────────────────────────────────

"""
Dataset for attack/evaluation.
"""
struct Dataset
    samples::Vector{Array{Float32}}
    labels::Vector{Int}
    name::String
    num_classes::Int
end

function Dataset(samples::Vector, labels::Vector; name::String = "unnamed")
    Dataset(
        [Float32.(s) for s in samples],
        labels,
        name,
        length(unique(labels))
    )
end

"""
Batch iterator.
"""
struct BatchIterator
    dataset::Dataset
    batch_size::Int
    shuffle::Bool
end

function Base.iterate(iter::BatchIterator, state = 1)
    state > length(iter.dataset.samples) && return nothing
    
    end_idx = min(state + iter.batch_size - 1, length(iter.dataset.samples))
    batch_samples = iter.dataset.samples[state:end_idx]
    batch_labels = iter.dataset.labels[state:end_idx]
    
    return ((batch_samples, batch_labels), end_idx + 1)
end

Base.length(iter::BatchIterator) = ceil(Int, length(iter.dataset.samples) / iter.batch_size)
