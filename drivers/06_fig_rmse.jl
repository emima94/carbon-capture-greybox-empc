# ------------------------------------------------------------------------------------------
# Figure 5: Out-of-sample RMSE, phi_oos, vs. prediction horizon (lambda_reg = 0.1)
# Figure 6: Out-of-sample RMSE, phi_oos, and in-sample RMSE, phi_is, vs. NN regularization
#           parameter lambda_reg (prediction horizon: 1 h)
#
# Requires: results/predictions.jld2 (driver 04)
# Output:   figures/fig05a_rmse_yg_vs_horizon.pdf, figures/fig05b_rmse_Q_vs_horizon.pdf,
#           figures/fig06a_rmse_yg_vs_lambda.pdf,  figures/fig06b_rmse_Q_vs_lambda.pdf
# ------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
set_plot_defaults(guidefont = Plots.font(9, "Times New Roman"), fontfamily = "Times New Roman")

predictions = load(joinpath(RESULTS_DIR, "predictions.jld2"), "predictions")
prediction_horizons = load(joinpath(RESULTS_DIR, "predictions.jld2"), "prediction_horizons")

# Cross-validated RMSE across series, phi = sqrt(mean(phi_m^2))
rms(x) = sqrt(mean(x .^ 2))
select(log10_lambda, sample_type) = filter(r -> r.log10_lambda == log10_lambda && r.sample_type == sample_type, predictions)
phi(log10_lambda, sample_type, var, h) = rms([getproperty(r, var)[h] for r in select(log10_lambda, sample_type)])

ms = 2.5
w, h, s = 225, 165, 1.10

## Figure 5: RMSE vs. prediction horizon
x_lbl = latexstring("\\textrm{Prediction\\ horizon\\ [h]}")
xlim_plt = (0, 4.0)
horizon_idxs = eachindex(prediction_horizons)

plt_yga = plot(ylims = (0.0, 0.8),
               ylabel = latexstring("\\textrm{RMSE,\\ } y_g \\textrm{\\ [\\%vol]}"),
               xlabel = x_lbl,
               xlim = xlim_plt)
plt_Q = plot(ylims = (0.0, 3.0),
             ylabel = latexstring("\\textrm{RMSE,\\ } Q \\textrm{\\ [kW]}"),
             xlabel = x_lbl,
             xlim = xlim_plt)

plot!(plt_yga, prediction_horizons, [phi(LOG10_LAMBDA_CHOSEN, "out-of-sample", :RMSE_yg, k) for k in horizon_idxs],
      label = "", marker = :circle, lw = 2, markersize = ms, markerstrokewidth = 0.0, color = colors_palette.blue)
plot!(plt_Q, prediction_horizons, [phi(LOG10_LAMBDA_CHOSEN, "out-of-sample", :RMSE_Q, k) for k in horizon_idxs],
      label = "EKF", marker = :circle, lw = 2, markersize = ms, markerstrokewidth = 0.0, color = colors_palette.blue)
plot!(plt_Q, legend = false)

for (plt, name) in [(plt_yga, "fig05a_rmse_yg_vs_horizon.pdf"), (plt_Q, "fig05b_rmse_Q_vs_horizon.pdf")]
    savefig(plot!(plt, layout = (1, 1), size = (w, h) .* s, bottom_margin = -2mm, left_margin = 0mm, right_margin = 0mm),
            joinpath(FIG_DIR, name))
end

## Figure 6: RMSE vs. lambda_reg, 1 h ahead predictions
x_lbl = latexstring("\\textrm{NN\\ regularization,}\\ \\log_{10}(\\lambda_{\\mathrm{reg}})")
colors_is_oos = [colors_palette.blue, colors_palette.orange]
lgds = (is = latexstring("\\textrm{in\u2010sample,\\ } \\varphi_{is}"),
        oos = latexstring("\\textrm{out\u2010of\u2010sample,\\ } \\varphi_{oos}"))
h_1h = findfirst(==(1.0), prediction_horizons)

p_yga = plot(xlabel = x_lbl, ylabel = latexstring("\\textrm{RMSE,}\\ y_g \\textrm{\\ [\\%vol]}"), ylims = (0, 1.4))
p_Q = plot(xlabel = x_lbl, ylabel = latexstring("\\textrm{RMSE,}\\ Q \\textrm{\\ [kW]}"), ylims = (0, 5.0))

for (p, var) in [(p_yga, :RMSE_yg), (p_Q, :RMSE_Q)]
    for (sample_type, lbl, color) in [("in-sample", lgds.is, colors_is_oos[1]), ("out-of-sample", lgds.oos, colors_is_oos[2])]
        plot!(p, LOG10_LAMBDA_VALUES, [phi(l, sample_type, var, h_1h) for l in LOG10_LAMBDA_VALUES],
              label = lbl, marker = :circle, lw = 2, markersize = ms, markerstrokewidth = 0.0, color = color)
    end
end
plot!(p_Q, legend = false)

for (plt, name) in [(p_yga, "fig06a_rmse_yg_vs_lambda.pdf"), (p_Q, "fig06b_rmse_Q_vs_lambda.pdf")]
    savefig(plot!(plt, layout = (1, 1), size = (w, h) .* s, bottom_margin = -2mm, left_margin = 0mm, right_margin = -20mm),
            joinpath(FIG_DIR, name))
end

println("Saved figures/fig05a_rmse_yg_vs_horizon.pdf, fig05b_rmse_Q_vs_horizon.pdf, fig06a_rmse_yg_vs_lambda.pdf, fig06b_rmse_Q_vs_lambda.pdf")
