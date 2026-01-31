# ═══════════════════════════════════════════════════════════════════════════════
#                              MIRAGE - Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
#                              PATHS
# ───────────────────────────────────────────────────────────────────────────────

const CONFIG_PATHS = Dict{Symbol, String}(
    :config_dir => expanduser("~/.config/mirage"),
    :config_file => expanduser("~/.config/mirage/config.toml"),
    :cache_dir => expanduser("~/.cache/mirage"),
    :models_dir => expanduser("~/.cache/mirage/models"),
    :logs_dir => expanduser("~/.cache/mirage/logs"),
)

# ───────────────────────────────────────────────────────────────────────────────
#                              DEFAULT CONFIG
# ───────────────────────────────────────────────────────────────────────────────

const DEFAULT_CONFIG = Dict{Symbol, Any}(
    :device => :cpu,
    :threads => Threads.nthreads(),
    :precision => Float32,
    :verbose => true,
    :log_queries => true,
    :cache_gradients => true,
    
    # Attack defaults
    :default_epsilon => 0.03,
    :default_norm => :linf,
    :default_iterations => 100,
    :default_step_size => 0.01,
    
    # Black-box defaults
    :max_queries => 10000,
    :batch_size => 64,
    
    # Extraction defaults
    :extraction_budget => 50000,
    :surrogate_architecture => :resnet18,
    
    # Logging
    :log_level => :info,
    :log_to_file => false,
)

# ───────────────────────────────────────────────────────────────────────────────
#                              FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

"""
    load_config()

Load configuration from file or return defaults.
"""
function load_config()::Dict{Symbol, Any}
    config = copy(DEFAULT_CONFIG)
    
    config_file = CONFIG_PATHS[:config_file]
    
    if isfile(config_file)
        try
            file_config = TOML.parsefile(config_file)
            
            for (key, value) in file_config
                config[Symbol(key)] = value
            end
        catch e
            @warn "Failed to load config file: $e"
        end
    end
    
    # Environment variable overrides
    haskey(ENV, "MIRAGE_DEVICE") && (config[:device] = Symbol(ENV["MIRAGE_DEVICE"]))
    haskey(ENV, "MIRAGE_THREADS") && (config[:threads] = parse(Int, ENV["MIRAGE_THREADS"]))
    haskey(ENV, "MIRAGE_VERBOSE") && (config[:verbose] = ENV["MIRAGE_VERBOSE"] == "true")
    
    return config
end

"""
    save_config(config::Dict{Symbol, Any})

Save configuration to file.
"""
function save_config(config::Dict{Symbol, Any})
    config_dir = CONFIG_PATHS[:config_dir]
    config_file = CONFIG_PATHS[:config_file]
    
    !isdir(config_dir) && mkpath(config_dir)
    
    # Convert to string keys for TOML
    str_config = Dict{String, Any}()
    for (k, v) in config
        str_config[string(k)] = v isa Symbol ? string(v) : v
    end
    
    open(config_file, "w") do io
        TOML.print(io, str_config)
    end
end

"""
    ensure_directories()

Create necessary directories.
"""
function ensure_directories()
    for (_, path) in CONFIG_PATHS
        if !endswith(path, ".toml") && !isdir(path)
            mkpath(path)
        end
    end
end

"""
    get_cache_path(name::String)

Get path for cached item.
"""
function get_cache_path(name::String)::String
    joinpath(CONFIG_PATHS[:cache_dir], name)
end

"""
    get_model_path(name::String)

Get path for model.
"""
function get_model_path(name::String)::String
    joinpath(CONFIG_PATHS[:models_dir], name)
end

# ───────────────────────────────────────────────────────────────────────────────
#                              ATTACK PRESETS
# ───────────────────────────────────────────────────────────────────────────────

const ATTACK_PRESETS = Dict{Symbol, AttackConfig}(
    :fast => AttackConfig(
        norm = Linf,
        epsilon = 0.03,
        targeted = false,
        target_class = nothing,
        confidence = 0.0,
        max_iterations = 10,
        early_stop = true
    ),
    :balanced => AttackConfig(
        norm = Linf,
        epsilon = 0.03,
        targeted = false,
        target_class = nothing,
        confidence = 0.0,
        max_iterations = 40,
        early_stop = true
    ),
    :strong => AttackConfig(
        norm = Linf,
        epsilon = 0.03,
        targeted = false,
        target_class = nothing,
        confidence = 5.0,
        max_iterations = 100,
        early_stop = false
    ),
    :stealth => AttackConfig(
        norm = L2,
        epsilon = 0.5,
        targeted = false,
        target_class = nothing,
        confidence = 0.0,
        max_iterations = 50,
        early_stop = true
    ),
)

"""
    get_preset(name::Symbol)

Get attack preset configuration.
"""
function get_preset(name::Symbol)::AttackConfig
    haskey(ATTACK_PRESETS, name) || error("Unknown preset: $name")
    return ATTACK_PRESETS[name]
end

# ───────────────────────────────────────────────────────────────────────────────
#                              EPSILON GUIDELINES
# ───────────────────────────────────────────────────────────────────────────────

const EPSILON_GUIDELINES = Dict{Tuple{NormType, Symbol}, Float64}(
    # ImageNet scale (0-1)
    (Linf, :imagenet) => 4/255,    # ~0.0157
    (L2, :imagenet) => 3.0,
    (L0, :imagenet) => 100.0,
    
    # CIFAR scale (0-1)
    (Linf, :cifar) => 8/255,       # ~0.0314
    (L2, :cifar) => 0.5,
    (L0, :cifar) => 30.0,
    
    # MNIST scale (0-1)
    (Linf, :mnist) => 0.3,
    (L2, :mnist) => 2.0,
    (L0, :mnist) => 20.0,
)

"""
    recommended_epsilon(norm::NormType, dataset::Symbol)

Get recommended epsilon for dataset.
"""
function recommended_epsilon(norm::NormType, dataset::Symbol)::Float64
    key = (norm, dataset)
    haskey(EPSILON_GUIDELINES, key) || return 0.03  # Default
    return EPSILON_GUIDELINES[key]
end
