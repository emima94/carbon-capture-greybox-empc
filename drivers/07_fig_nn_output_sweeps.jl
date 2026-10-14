# ------------------------------------------------------------------------------------------
# Figure 7: NN outputs (desorption rate Nd and reboiler duty Q) for each out-of-sample
# parameter estimate, for lambda_reg = 0.0001 (blue) and lambda_reg = 0.1 (orange), swept
# over the solvent flow rate F and the desorber temperature set point Td_sp.
#
# Output: figures/fig07a_nn_output_sweep_Nd.pdf, figures/fig07b_nn_output_sweep_Q.pdf
# ------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
set_plot_defaults(guidefont = Plots.font(9, "Times New Roman"), fontfamily = "Times New Roman")

log10_lambda_sweep = [-4.0, -1.0]

# Input ranges and fixed values of the other NN inputs
u_min = [0.20, 85.0]          # [F, Td_sp]
u_max = [0.80, 95.0]
ca_mean = 2.60
cd_mean = 2.25
Tla_mean = 37.50
N = 50
u_range = [range(u_min[i], u_max[i], length = N) for i in eachindex(u_min)]
u1_mean = mean(u_range[1])
u2_mean = mean(u_range[2])

# Evaluate Nd and Q along the two input slices for each out-of-sample parameter estimate.
# The desorber temperature state is set equal to the set point, x = [ca, cd, Td_sp].
function sweep_outputs(p, hfun, p_fixed)
    Nd_u1, Q_u1, Nd_u2, Q_u2 = Float64[], Float64[], Float64[], Float64[]
    for u1 in u_range[1]
        out = named_output(hfun([ca_mean, cd_mean, u2_mean], [u1, Tla_mean, u2_mean], [NaN, NaN], p, p_fixed))
        push!(Nd_u1, out.Nd[1]); push!(Q_u1, out.Q[1])
    end
    for u2 in u_range[2]
        out = named_output(hfun([ca_mean, cd_mean, u2], [u1_mean, Tla_mean, u2], [NaN, NaN], p, p_fixed))
        push!(Nd_u2, out.Nd[1]); push!(Q_u2, out.Q[1])
    end
    return (Nd_u1 = Nd_u1, Q_u1 = Q_u1, Nd_u2 = Nd_u2, Q_u2 = Q_u2)
end

## Plot
xlbl_F = latexstring("\\textrm{Solvent\\ flow\\ rate,} F, \\ \\textrm{[m}^3\\textrm{/h]}")
xlbl_Tld = latexstring("\\textrm{Desorber\\ temperature\\ SP,\\ } T^{sp}_d, \\ \\textrm{[°C]}")
ylbl_Nd = latexstring("\\textrm{Desorption\\ rate,}") * "\n" * latexstring("N_d, \\ \\textrm{[mol/h]}")
ylbl_Q = latexstring("\\textrm{Reboiler\\ duty,}") * "\n" * latexstring("Q, \\ \\textrm{[kW]}")
ylims_Nd = (0, 500.0)
ylims_Q = (0, 45.0)
colors_lambda = [colors_palette.blue, colors_palette.orange]
lw = 1.0

p_Nd_u1 = plot(xlabel = xlbl_F, ylabel = ylbl_Nd, ylims = ylims_Nd)
p_Q_u1 = plot(xlabel = xlbl_F, ylabel = ylbl_Q, ylims = ylims_Q)
p_Nd_u2 = plot(xlabel = xlbl_Tld, ylabel = ylbl_Nd, ylims = ylims_Nd)
p_Q_u2 = plot(xlabel = xlbl_Tld, ylabel = ylbl_Q, ylims = ylims_Q)

for (k, log10_lambda) in enumerate(log10_lambda_sweep)
    color = mix_with_white(colors_lambda[k], 0.0)
    estimates = filter(e -> e.sample_type == "out-of-sample", load_parameter_estimates(log10_lambda))
    model = build_dti_model(estimates[1].lambda_reg)

    # Legend entry
    plot!(p_Nd_u2, [], [], label = latexstring("\\lambda_{reg} = ") * string(10.0^log10_lambda), linewidth = 2.5, color = color)

    for est in estimates
        out = sweep_outputs(est.p, model.hfun, model.p_fixed)
        plot!(p_Nd_u1, u_range[1], out.Nd_u1, label = "", linewidth = lw, color = color)
        plot!(p_Q_u1, u_range[1], out.Q_u1, label = "", linewidth = lw, color = color)
        plot!(p_Nd_u2, u_range[2], out.Nd_u2, label = "", linewidth = lw, color = color)
        plot!(p_Q_u2, u_range[2], out.Q_u2, label = "", linewidth = lw, color = color)
    end
end

# Margins: y-axis label and tick labels only on the left plot
for plt in [p_Nd_u1, p_Q_u1]
    plot!(plt, left_margin = 8mm, bottom_margin = 5mm)
end
for plt in [p_Nd_u2, p_Q_u2]
    ytick_i = yticks(plt)
    plot!(plt, yticks = (ytick_i[1][1], ["" for _ in 1:length(ytick_i[1][1])]), ylabel = "")
    plot!(plt, left_margin = -6mm, bottom_margin = 5mm)
end

savefig(plot(p_Nd_u1, p_Nd_u2, layout = (1, 2), size = 0.85 .* (600, 250)), joinpath(FIG_DIR, "fig07a_nn_output_sweep_Nd.pdf"))
savefig(plot(p_Q_u1, p_Q_u2, layout = (1, 2), size = 0.85 .* (600, 250)), joinpath(FIG_DIR, "fig07b_nn_output_sweep_Q.pdf"))
println("Saved figures/fig07a_nn_output_sweep_Nd.pdf, fig07b_nn_output_sweep_Q.pdf")
