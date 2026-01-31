# ═══════════════════════════════════════════════════════════════════════════════
#                              MIRAGE - Display
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
#                              BOX CHARACTERS
# ───────────────────────────────────────────────────────────────────────────────

const BOX = (
    tl = "╔", tr = "╗", bl = "╚", br = "╝",
    h = "═", v = "║",
    lt = "╠", rt = "╣", tt = "╦", bt = "╩",
    cross = "╬",
    light_h = "─", light_v = "│",
)

# ───────────────────────────────────────────────────────────────────────────────
#                              ICONS
# ───────────────────────────────────────────────────────────────────────────────

const ICONS = Dict(
    :success => "✓",
    :failure => "✗",
    :warning => "⚠",
    :info => "ℹ",
    :attack => "⚔",
    :defense => "🛡",
    :target => "🎯",
    :model => "🧠",
    :query => "❓",
    :gradient => "∇",
    :perturbation => "δ",
    :extraction => "📥",
    :analysis => "🔬",
)

# ───────────────────────────────────────────────────────────────────────────────
#                              PROGRESS
# ───────────────────────────────────────────────────────────────────────────────

const PROGRESS_CHARS = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

"""
    progress_bar(progress::Float64, width::Int = 30)

Create a progress bar string.
"""
function progress_bar(progress::Float64, width::Int = 30)::String
    progress = clamp(progress, 0.0, 1.0)
    filled = floor(Int, progress * width)
    partial_idx = floor(Int, (progress * width - filled) * 8) + 1
    partial_idx = clamp(partial_idx, 1, 8)
    
    bar = "█"^filled
    if filled < width
        bar *= string(PROGRESS_CHARS[partial_idx])
        bar *= " "^(width - filled - 1)
    end
    
    return "[$(bar)]"
end

"""
    spinner(frame::Int)

Get spinner frame.
"""
spinner(frame::Int) = SPINNER_FRAMES[mod1(frame, length(SPINNER_FRAMES))]

# ───────────────────────────────────────────────────────────────────────────────
#                              ATTACK RESULT DISPLAY
# ───────────────────────────────────────────────────────────────────────────────

"""
    display_result(result::AttackResult)

Display attack result.
"""
function display_result(result::AttackResult)
    status_color = result.status == SUCCESS ? :green : 
                   result.status == PARTIAL ? :yellow : :red
    status_icon = result.status == SUCCESS ? ICONS[:success] : ICONS[:failure]
    
    println()
    println(c(:magenta), BOX.tl, BOX.h^61, BOX.tr, c(:reset))
    println(c(:magenta), BOX.v, c(:reset), "                     ATTACK RESULT                          ", c(:magenta), BOX.v, c(:reset))
    println(c(:magenta), BOX.lt, BOX.h^61, BOX.rt, c(:reset))
    
    println(c(:magenta), BOX.v, c(:reset), " Status:      ", c(status_color), status_icon, " ", result.status, c(:reset))
    println(c(:magenta), BOX.v, c(:reset), " Original:    ", c(:cyan), "Class $(result.original_label) ($(round(result.confidence_original * 100, digits=1))%)", c(:reset))
    println(c(:magenta), BOX.v, c(:reset), " Adversarial: ", c(:yellow), "Class $(result.adversarial_label) ($(round(result.confidence_adversarial * 100, digits=1))%)", c(:reset))
    
    println(c(:magenta), BOX.lt, BOX.h^61, BOX.rt, c(:reset))
    println(c(:magenta), BOX.v, c(:reset), " Perturbation (L2):   ", c(:white), @sprintf("%.6f", result.perturbation_l2), c(:reset))
    println(c(:magenta), BOX.v, c(:reset), " Perturbation (L∞):   ", c(:white), @sprintf("%.6f", result.perturbation_linf), c(:reset))
    println(c(:magenta), BOX.v, c(:reset), " Queries Used:        ", c(:white), "$(result.queries_used)", c(:reset))
    println(c(:magenta), BOX.v, c(:reset), " Iterations:          ", c(:white), "$(result.iterations)", c(:reset))
    println(c(:magenta), BOX.v, c(:reset), " Time Elapsed:        ", c(:dim), @sprintf("%.2fs", result.time_elapsed), c(:reset))
    
    println(c(:magenta), BOX.bl, BOX.h^61, BOX.br, c(:reset))
end

"""
    display_attack_progress(iteration::Int, max_iter::Int, loss::Float64, success::Bool)

Display attack progress.
"""
function display_attack_progress(iteration::Int, max_iter::Int, loss::Float64, success::Bool)
    progress = iteration / max_iter
    bar = progress_bar(progress, 25)
    
    status = success ? "$(c(:green))$(ICONS[:success])" : "$(c(:yellow))..."
    
    print("\r  $(c(:dim))Iteration $(c(:white))$iteration/$max_iter $(c(:dim))$bar $(c(:cyan))Loss: $(@sprintf("%.4f", loss)) $status$(c(:reset))    ")
end

# ───────────────────────────────────────────────────────────────────────────────
#                              ROBUSTNESS REPORT
# ───────────────────────────────────────────────────────────────────────────────

"""
    display_report(results::Vector{RobustnessResult})

Display robustness evaluation report.
"""
function display_report(results::Vector{RobustnessResult})
    println()
    println(c(:magenta), BOX.tl, BOX.h^75, BOX.tr, c(:reset))
    println(c(:magenta), BOX.v, c(:reset), "                         ROBUSTNESS REPORT                                ", c(:magenta), BOX.v, c(:reset))
    println(c(:magenta), BOX.lt, BOX.h^75, BOX.rt, c(:reset))
    
    # Header
    println(c(:magenta), BOX.v, c(:reset), 
        c(:bold), " Attack         ", c(:reset), "│",
        c(:bold), " ε       ", c(:reset), "│",
        c(:bold), " Success % ", c(:reset), "│",
        c(:bold), " Avg L2    ", c(:reset), "│",
        c(:bold), " Queries   ", c(:reset),
        c(:magenta), BOX.v, c(:reset))
    println(c(:magenta), BOX.v, c(:reset), "─"^16, "┼", "─"^9, "┼", "─"^11, "┼", "─"^11, "┼", "─"^11, c(:magenta), BOX.v, c(:reset))
    
    for r in results
        success_color = r.success_rate > 0.8 ? :red : r.success_rate > 0.5 ? :yellow : :green
        
        attack_str = rpad(string(r.attack_name), 15)
        eps_str = rpad(@sprintf("%.4f", r.epsilon), 8)
        success_str = rpad(@sprintf("%.1f%%", r.success_rate * 100), 10)
        pert_str = rpad(@sprintf("%.4f", r.avg_perturbation), 10)
        queries_str = rpad(@sprintf("%.0f", r.avg_queries), 10)
        
        println(c(:magenta), BOX.v, c(:reset), 
            " ", c(:cyan), attack_str, c(:reset), "│",
            " ", c(:white), eps_str, c(:reset), "│",
            " ", c(success_color), success_str, c(:reset), "│",
            " ", c(:dim), pert_str, c(:reset), "│",
            " ", c(:dim), queries_str, c(:reset),
            c(:magenta), BOX.v, c(:reset))
    end
    
    println(c(:magenta), BOX.bl, BOX.h^75, BOX.br, c(:reset))
end

# ───────────────────────────────────────────────────────────────────────────────
#                              EXTRACTION DISPLAY
# ───────────────────────────────────────────────────────────────────────────────

"""
    display_extraction_progress(queries::Int, budget::Int, fidelity::Float64)

Display extraction progress.
"""
function display_extraction_progress(queries::Int, budget::Int, fidelity::Float64)
    progress = queries / budget
    bar = progress_bar(progress, 25)
    
    fidelity_color = fidelity > 0.9 ? :green : fidelity > 0.7 ? :yellow : :red
    
    print("\r  $(c(:dim))Queries $(c(:white))$queries/$budget $(c(:dim))$bar $(c(:cyan))Fidelity: $(c(fidelity_color))$(@sprintf("%.1f%%", fidelity * 100))$(c(:reset))    ")
end

"""
    display_extraction_result(metrics::ExtractionMetrics)

Display extraction result.
"""
function display_extraction_result(metrics::ExtractionMetrics)
    fidelity_color = metrics.fidelity > 0.9 ? :green : metrics.fidelity > 0.7 ? :yellow : :red
    
    println()
    println(c(:magenta), BOX.tl, BOX.h^61, BOX.tr, c(:reset))
    println(c(:magenta), BOX.v, c(:reset), "                  EXTRACTION RESULT                         ", c(:magenta), BOX.v, c(:reset))
    println(c(:magenta), BOX.lt, BOX.h^61, BOX.rt, c(:reset))
    
    println(c(:magenta), BOX.v, c(:reset), " Fidelity:        ", c(fidelity_color), @sprintf("%.2f%%", metrics.fidelity * 100), c(:reset))
    println(c(:magenta), BOX.v, c(:reset), " Test Accuracy:   ", c(:cyan), @sprintf("%.2f%%", metrics.accuracy * 100), c(:reset))
    println(c(:magenta), BOX.v, c(:reset), " Queries Used:    ", c(:white), "$(metrics.queries_used)", c(:reset))
    println(c(:magenta), BOX.v, c(:reset), " Training Time:   ", c(:dim), @sprintf("%.2fs", metrics.training_time), c(:reset))
    println(c(:magenta), BOX.v, c(:reset), " Model Size:      ", c(:dim), "$(metrics.model_size) params", c(:reset))
    
    println(c(:magenta), BOX.bl, BOX.h^61, BOX.br, c(:reset))
end

# ───────────────────────────────────────────────────────────────────────────────
#                              LOGGING
# ───────────────────────────────────────────────────────────────────────────────

"""
    log_info(msg::String)
"""
function log_info(msg::String)
    CONFIG[].verbose && println(c(:cyan), ICONS[:info], c(:reset), " ", msg)
end

"""
    log_success(msg::String)
"""
function log_success(msg::String)
    CONFIG[].verbose && println(c(:green), ICONS[:success], c(:reset), " ", msg)
end

"""
    log_warning(msg::String)
"""
function log_warning(msg::String)
    CONFIG[].verbose && println(c(:yellow), ICONS[:warning], c(:reset), " ", msg)
end

"""
    log_error(msg::String)
"""
function log_error(msg::String)
    println(c(:red), ICONS[:failure], c(:reset), " ", msg)
end

"""
    log_attack(name::Symbol, target::String)
"""
function log_attack(name::Symbol, target::String)
    CONFIG[].verbose && println(c(:magenta), ICONS[:attack], c(:reset), " Starting ", c(:cyan), name, c(:reset), " attack on ", c(:yellow), target, c(:reset))
end
