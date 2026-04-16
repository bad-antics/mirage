<div align="center">

# 🔮 MIRAGE

### Adversarial Machine Learning Toolkit

**Model Extraction • Adversarial Examples • Neural Network Probing**

[![Julia](https://img.shields.io/badge/Julia-1.10+-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://julialang.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![NullSec](https://img.shields.io/badge/NullSec-Integration-red?style=for-the-badge)](https://github.com/bad-antics/nullsec-linux)

```
# Mirage — Adversarial ML Security Suite

    ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
                                                    
      [ Adversarial ML Toolkit | bad-antics ]
```

</div>

---

## ⚡ Overview

**Mirage** is a high-performance adversarial machine learning toolkit written in Julia, designed for security researchers and red teamers to evaluate ML model robustness. It provides tools for:

- **Model Extraction** — Steal model functionality through query-based attacks
- **Adversarial Examples** — Generate inputs that fool classifiers
- **Neural Network Probing** — Analyze model internals and decision boundaries
- **Defense Evaluation** — Test robustness of defensive measures

## 🎯 Features

### 🕵️ Model Extraction Attacks
| Attack | Description |
|--------|-------------|
| **Query Synthesis** | Generate synthetic queries to extract decision boundaries |
| **Knockoff Nets** | Train surrogate models using API queries |
| **JBDA** | Jacobian-Based Dataset Augmentation |
| **ActiveThief** | Active learning for efficient extraction |
| **CloudLeak** | MLaaS-specific extraction techniques |

### 💥 Adversarial Example Generation
| Method | Type | Description |
|--------|------|-------------|
| **FGSM** | White-box | Fast Gradient Sign Method |
| **PGD** | White-box | Projected Gradient Descent |
| **C&W** | White-box | Carlini-Wagner L2/L∞ attack |
| **DeepFool** | White-box | Minimal perturbation finder |
| **AutoAttack** | White-box | Ensemble of strongest attacks |
| **Square** | Black-box | Query-efficient score-based |
| **HopSkipJump** | Black-box | Decision-based attack |
| **Boundary** | Black-box | Decision boundary attack |
| **SimBA** | Black-box | Simple Black-box Attack |
| **QEBA** | Black-box | Query-Efficient Boundary Attack |

### 🔬 Neural Network Analysis
- **Gradient Saliency** — Visualize input importance
- **Integrated Gradients** — Attribution methods
- **LIME** — Local interpretable explanations
- **Decision Boundary Mapping** — 2D/3D visualization
- **Neuron Activation Analysis** — Internal representation probing
- **Layer-wise Relevance Propagation** — Contribution analysis

### 🛡️ Defense Testing
- **Adversarial Training Evaluation**
- **Input Preprocessing Bypass**
- **Certified Defense Verification**
- **Ensemble Robustness Testing**
- **Detection Evasion**

---

## 🚀 Quick Start

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/bad-antics/mirage")

# Or clone and develop
Pkg.develop(path="/path/to/mirage")
```

### Basic Usage

```julia
using Mirage

# Initialize
Mirage.banner()
config = Mirage.init()

# ═══════════════════════════════════════════════════════════════════
# Model Extraction Attack
# ═══════════════════════════════════════════════════════════════════

# Define target API
target = RemoteModel(
    "https://api.target.com/predict",
    headers = Dict("Authorization" => "Bearer TOKEN")
)

# Extract model
surrogate = extract_model(target,
    method = :knockoff,
    budget = 10000,          # Max queries
    input_shape = (28, 28),
    num_classes = 10
)

# Test fidelity
fidelity = evaluate_fidelity(surrogate, target, test_samples)
println("Extraction fidelity: $(fidelity * 100)%")

# ═══════════════════════════════════════════════════════════════════
# Adversarial Example Generation
# ═══════════════════════════════════════════════════════════════════

# Load local model
model = load_model("classifier.onnx")

# Generate adversarial examples
adversarial = attack(model, image,
    method = :pgd,
    epsilon = 0.03,
    iterations = 40,
    step_size = 0.01
)

# Black-box attack (only predictions available)
adversarial = attack(target, image,
    method = :square,
    epsilon = 0.05,
    max_queries = 5000
)

# Check success
original_pred = predict(model, image)
adv_pred = predict(model, adversarial)
println("Original: $original_pred → Adversarial: $adv_pred")

# ═══════════════════════════════════════════════════════════════════
# Model Analysis
# ═══════════════════════════════════════════════════════════════════

# Gradient saliency
saliency = gradient_saliency(model, image)
visualize_saliency(saliency)

# Decision boundary
boundary = map_decision_boundary(model, 
    samples = test_data,
    resolution = 100
)
plot_boundary(boundary)

# Neuron analysis
activations = probe_neurons(model, image, layer = 5)
top_neurons = most_activated(activations, k = 10)
```

---

## 📖 Attack Reference

### White-Box Attacks

```julia
# FGSM - Fast Gradient Sign Method
adv = fgsm(model, x, y, epsilon = 0.03)

# PGD - Projected Gradient Descent
adv = pgd(model, x, y,
    epsilon = 0.03,
    alpha = 0.01,
    iterations = 40,
    random_start = true
)

# C&W - Carlini-Wagner
adv = carlini_wagner(model, x, y,
    confidence = 0.0,
    learning_rate = 0.01,
    max_iterations = 1000,
    binary_search_steps = 9
)

# DeepFool
adv = deepfool(model, x,
    max_iterations = 50,
    overshoot = 0.02
)

# AutoAttack (strongest combination)
adv = auto_attack(model, x, y,
    epsilon = 8/255,
    attacks = [:apgd_ce, :apgd_dlr, :fab, :square]
)
```

### Black-Box Attacks

```julia
# Square Attack (score-based)
adv = square_attack(model, x, y,
    epsilon = 0.05,
    max_queries = 5000,
    p_init = 0.8
)

# HopSkipJump (decision-based)
adv = hopskipjump(model, x,
    target_label = nothing,  # untargeted
    max_queries = 10000,
    gamma = 1.0
)

# Boundary Attack
adv = boundary_attack(model, x,
    max_iterations = 10000,
    spherical_step = 0.01,
    source_step = 0.01
)

# SimBA
adv = simba(model, x, y,
    epsilon = 0.2,
    max_queries = 10000,
    freq_dims = 28  # DCT basis
)
```

### Model Extraction

```julia
# Knockoff Nets
surrogate = knockoff_nets(target,
    architecture = :resnet18,
    budget = 50000,
    batch_size = 256
)

# JBDA - Jacobian-Based Data Augmentation
surrogate = jbda_extract(target,
    seed_data = initial_samples,
    augmentation_factor = 10,
    budget = 20000
)

# Active Learning Extraction
surrogate = active_thief(target,
    strategy = :entropy,
    budget = 10000,
    batch_size = 100
)
```

---

## 🏗️ Architecture

```
mirage/
├── src/
│   ├── Mirage.jl           # Main module
│   ├── core/
│   │   ├── Types.jl        # Type definitions
│   │   ├── Config.jl       # Configuration
│   │   ├── Display.jl      # Terminal UI
│   │   └── Utils.jl        # Utilities
│   ├── attacks/
│   │   ├── WhiteBox.jl     # White-box attacks
│   │   ├── BlackBox.jl     # Black-box attacks
│   │   └── Extraction.jl   # Model extraction
│   ├── models/
│   │   ├── Loaders.jl      # Model loading
│   │   ├── Remote.jl       # Remote API interface
│   │   └── Surrogate.jl    # Surrogate training
│   ├── analysis/
│   │   ├── Saliency.jl     # Attribution methods
│   │   ├── Probing.jl      # Network probing
│   │   └── Boundary.jl     # Decision boundaries
│   └── defenses/
│       ├── Detection.jl    # Attack detection
│       └── Evaluation.jl   # Defense testing
├── test/
├── examples/
└── docs/
```

---

## 🔧 Configuration

```julia
# Configure Mirage
Mirage.configure(
    device = :cuda,           # :cpu, :cuda, :metal
    threads = 8,
    precision = Float32,
    verbose = true,
    log_queries = true,
    cache_gradients = true
)

# Attack-specific config
attack_config = AttackConfig(
    norm = :linf,             # :l2, :linf, :l0, :l1
    epsilon = 0.03,
    targeted = false,
    confidence = 0.0
)
```

---

## 📊 Metrics & Evaluation

```julia
# Evaluate attack success
metrics = evaluate_attack(model, clean, adversarial, labels)

# Metrics include:
# - attack_success_rate
# - average_perturbation (L2, Linf)
# - queries_used
# - confidence_drop
# - transferability (to other models)

# Evaluate model robustness
robustness = evaluate_robustness(model, test_data,
    attacks = [:fgsm, :pgd, :square],
    epsilons = [0.01, 0.03, 0.05, 0.1]
)

# Generate robustness report
report = robustness_report(model, test_data)
display_report(report)
```

---

## 🔗 NullSec Integration

Mirage integrates seamlessly with NullSec Linux:

```julia
using Mirage

# Auto-detect NullSec environment
if Mirage.nullsec_available()
    # Use shared config
    Mirage.init_nullsec!()
    
    # Log attacks to NullSec
    Mirage.log_attack(result)
    
    # Access NullSec models
    models = Mirage.list_nullsec_models()
end
```

---

## ⚠️ Responsible Use

Mirage is designed for:
- ✅ Security research and model robustness evaluation
- ✅ Red team assessments with authorization
- ✅ Academic research on adversarial ML
- ✅ Testing your own models and defenses

**NOT** for:
- ❌ Attacking models without authorization
- ❌ Bypassing security in production systems
- ❌ Any malicious purpose

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 🔮 by [bad-antics](https://github.com/bad-antics)**

*Part of the NullSec Security Toolkit*

[![GitHub](https://img.shields.io/badge/GitHub-bad--antics-181717?style=flat-square&logo=github)](https://github.com/bad-antics)

</div>
