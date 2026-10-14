# ------------------------------------------------------------------------------------------
# Figure 4: Time series of experimental data and 1-hour ahead out-of-sample predictions on
# selected data series (lambda_reg = 0.1).
#
# Requires: results/predictions.jld2 (driver 04)
# Output:   figures/fig04_time_series_predictions.pdf
# ------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
set_plot_defaults(fontfamily = "Times New Roman", titlefontsize = 9, guidefontsize = 9)

datasets, _ = load_pilot_plant_data()
predictions = load(joinpath(RESULTS_DIR, "predictions.jld2"), "predictions")
prediction_horizons = load(joinpath(RESULTS_DIR, "predictions.jld2"), "prediction_horizons")

series_plt = [1, 4, 7, 8]              # Series shown in the paper
pred_horizon_idx = findfirst(==(1.0), prediction_horizons)
lw = 2.0

# Out-of-sample predictions for lambda_reg = 0.1: each series is predicted with the
# parameter estimate where that series was left out of the training data
oos_predictions = filter(r -> r.log10_lambda == LOG10_LAMBDA_CHOSEN && r.sample_type == "out-of-sample", predictions)

plot_list = Dict{Int, Any}()
for m in series_plt
    ds = datasets[m]
    pred = only(filter(r -> r.series == m, oos_predictions))

    plts = initialise_plots_paper(ds, "DTI")
    y_pred = (yga = pred.yg_pred[pred_horizon_idx], Q = pred.Q_pred[pred_horizon_idx], d_adaptive = nothing)
    add_to_plot_paper(plts, ds, y_pred, :solid, (:black, :black), "Prediction", adaptive = false, lw = lw)

    # Titles with series number and RMSE
    RMSE_yga = pred.RMSE_yg[pred_horizon_idx]
    RMSE_Q = pred.RMSE_Q[pred_horizon_idx]
    series_annotaion = latexstring("\\textrm{Series\\ }$(ds.series)")
    title_str_yga = series_annotaion *
                        "\n" *
                        latexstring("\\textrm{RMSE,}\\ y_g:\\ ") *
                        latexstring(@sprintf("%.2f", RMSE_yga) * "\\ \\textrm{vol}\\%")
    title_str_Q = latexstring("\\textrm{RMSE,}\\ Q:\\ ") *
                        latexstring(@sprintf("%.2f", RMSE_Q) * "\\ \\textrm{kW}")

    plot!(plts.plt_yga, top_margin = 10mm)
    plot!(plts.plt_Q, top_margin = 8mm)
    xmin_yga, xmax_yga = Plots.xlims(plts.plt_yga)
    ymin_yga, ymax_yga = Plots.ylims(plts.plt_yga)
    xmin_Q, xmax_Q = Plots.xlims(plts.plt_Q)
    ymin_Q, ymax_Q = Plots.ylims(plts.plt_Q)
    annotate!(plts.plt_yga, 0.5 * (xmax_yga - xmin_yga) + xmin_yga, 1.18 * (ymax_yga - ymin_yga) + ymin_yga, text(title_str_yga, font(9)))
    annotate!(plts.plt_Q, 0.5 * (xmax_Q - xmin_Q) + xmin_Q, 1.1 * (ymax_Q - ymin_Q) + ymin_Q, text(title_str_Q, font(9)))

    plot_list[m] = plts
end

# Combine: first row y_g, second row Q
plt_list_tot = [[plot_list[m].plt_yga for m in series_plt]..., [plot_list[m].plt_Q for m in series_plt]...]

left_margin_inner = -6mm
bottom_margin_inner = -5mm
left_margin_outer = 5mm
right_margin_outer = 2mm
bottom_margin_outer = 4mm
top_margin_outer = 8mm

for plt in plt_list_tot
    plot!(plt, left_margin = left_margin_inner, bottom_margin = bottom_margin_inner, right_margin = 0mm)
end

for (k, plt) in enumerate(plt_list_tot)
    # Legend only in the right-most plot of each row
    if k == 4
        plot!(plt, legend = (0.18, 0.85), legend_columns = 1)
    elseif k == 8
        plot!(plt, legend = (0.18, 0.8), legend_columns = 1)
    else
        plot!(plt, legend = false)
    end

    # y-axis label and tick labels only for the first plot in each row
    if !(k in [1, 5])
        ytick_i = yticks(plt)
        plot!(plt, yticks = (ytick_i[1][1], ["" for _ in 1:length(ytick_i[1][1])]), ylabel = "")
    else
        plot!(plt, left_margin = left_margin_outer)
    end

    # x-axis label and tick labels only for the bottom row
    if !(k in 5:8)
        xtick_i = xticks(plt)
        plot!(plt, xticks = (xtick_i[1][1], ["" for _ in 1:length(xtick_i[1][1])]), xlabel = "", top_margin = top_margin_outer)
    else
        plot!(plt, bottom_margin = bottom_margin_outer, xlabel = "Time [h]")
    end

    # Right-most margins
    if k in [4, 8]
        plot!(plt, right_margin = right_margin_outer)
    end
end

h = 450 * 2/3
s = 1.20
plt_final = plot(plt_list_tot..., layout = (2, 4), size = (800, h) .* s)
savefig(plt_final, joinpath(FIG_DIR, "fig04_time_series_predictions.pdf"))
println("Saved figures/fig04_time_series_predictions.pdf")
