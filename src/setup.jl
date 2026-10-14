# ------------------------------------------------------------------------------------------
# Common setup for the reproduction drivers: core setup (src/setup_core.jl) and plotting.
# ------------------------------------------------------------------------------------------

include("setup_core.jl")

using Plots
using LaTeXStrings
using Measures
using Colors

# Color palette used in the paper figures
const colors_palette = (red = "#FF0000",
                        blue = "#1f77b4",
                        lime = "#00FF00",
                        pink = "#FFC0CB",
                        cyan = "#00FFFF",
                        magenta = "#FF00FF",
                        orange = "#ff7f0e",
                        purple = "#9558B2",
                        green = "#008000",
                        darkgreen = "#008080")

"Reset the Plots.jl defaults and set the given defaults (each paper figure used its own defaults)."
function set_plot_defaults(; kwargs...)
    Plots.reset_defaults()
    gr()
    default(; kwargs...)
end
