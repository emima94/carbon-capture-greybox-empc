# ------------------------------------------------------------------------------------------
# Results of the EMPC demonstration at the DTI pilot plant (Section 7)
#
# Figure 12: Energy price profiles for the baseline and flexible operation scenarios
# Figure 13: Cumulative means of the outputs and the optimal inputs during the four tests
# Figure 14: Final outputs at the end of the four tests (bar charts, TikZ/pgfplots code)
# Text:      Final KPIs (capture efficiency, SRD, specific costs, cost reduction) and the
#            solve times of the OCP in the closed-loop tests
#
# Requires a LaTeX installation (PGFPlotsX compiles Figure 13 to pdf).
# Output: figures/fig12_energy_price.pdf, figures/fig13{a-f}_*.pdf,
#         figures/fig14_tex/kpi_{1-4}.tex (Fig. 14a = kpi_3, 14b = kpi_4, 14c = kpi_2),
#         results/empc_kpis.csv, results/ocp_solve_times.csv
# ------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
using PGFPlotsX
using TimeZones     # Needed to parse the time zone in the time stamps of the measurements
include(joinpath(@__DIR__, "..", "src", "closed_loop_data.jl"))
include(joinpath(@__DIR__, "..", "src", "util_closed_loop_MPC_plots.jl"))

push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usetikzlibrary{patterns}")
set_plot_defaults(fontfamily = "times", titlefontsize = 9, guidefontsize = 9)

tests_dir = joinpath(DATA_DIR, "closed_loop_tests")
tests = CSV.read(joinpath(tests_dir, "tests.csv"), DataFrame)

# Test settings
t0 = 0.0
tf = 6.0                                     # Test duration [h]
dt_control_interval = 10.0 / 60              # Control interval [h]
N_pel = Int64(tf / dt_control_interval)

# Energy price signal of the flexible scenario, p_k = 80 - 60 sin(2 pi t_k / 6) [EUR/MWh].
# All specific costs are evaluated with this price signal (also for the baseline scenarios).
p_el = 80 .- 60 .* sin.(2π .* (0:(N_pel-1)) * dt_control_interval / tf)
time_pel = collect((1:36) .- 1) * dt_control_interval

kW2MW = 1e-3
kWh2MJ = 3.6

## KPIs for each test
kpi_df = DataFrame(test = Int[], scenario = String[], cap_eff_final = Float64[], SRD_final = Float64[],
                   Q_mean = Float64[], costs_tot = Float64[], costs_per_ton_CO2 = Float64[])
test_data = []

for test in eachrow(tests)
    println("Evaluating test #$(test.test): $(test.folder)...")
    time_start = DateTime(test.time_start)
    time_final = time_start + Hour(tf)

    # Measurements in the test window
    df = load_csv_data_and_process(joinpath(tests_dir, test.folder, "measurements.csv"))
    idx_start = findfirst(df.time .>= time_start) - 1
    idx_final = findfirst(df.time .>= time_final) - 1
    df_test = df[idx_start:idx_final, :]

    time_test = [time_start; df_test.time[2:end]; time_final]
    t = (time_test .- time_start) ./ Hour(1)
    dt = diff(t)

    # KPIs at the end of the test
    Mga = sum(df_test.mga_CO2 .* dt)                       # Captured CO2 [kg]
    Mgina = sum(df_test.mgina_CO2 .* dt)                   # CO2 into the absorber [kg]
    cap_eff_final = Mga / Mgina * 100                      # Capture efficiency [%]
    Q_tot = sum(df_test.Q .* dt)                           # Reboiler energy [kWh]
    p_el_time_grid = [get_control(t[i], time_pel, p_el, 0.0) for i in eachindex(t[1:end-1])]
    costs = sum(p_el_time_grid .* df_test.Q * kW2MW .* dt)  # Energy costs [EUR]
    costs_per_ton_CO2 = costs / (Mga * 1e-3)               # Specific costs [EUR/ton CO2]
    SRD_final = Q_tot * kWh2MJ / Mga                       # SRD [MJ/kg CO2]

    # Applied inputs (set points)
    inputs = CSV.read(joinpath(tests_dir, test.folder, "inputs.csv"), DataFrame)
    u_opt = [[row.F_sp, row.Td_sp] for row in eachrow(inputs)]
    t_control = nrow(inputs) == 1 ? [t0, tf] : collect(t0:dt_control_interval:tf)

    scenario = "$(test.scenario), $(test.loop)-loop"
    push!(kpi_df, (test.test, scenario, cap_eff_final, SRD_final, Q_tot / tf, costs, costs_per_ton_CO2))
    push!(test_data, (t = t, df_test = df_test, p_el_time_grid = p_el_time_grid, t_control = t_control, u_opt = u_opt))
end

## Figure 13: cumulative means and optimal inputs (pgfplots)
lbls = ["\\#1, baseline, open-loop",
        "\\#2, flex. operation, open-loop",
        "\\#3, baseline, closed-loop",
        "\\#4, flex. operation, closed-loop"]
linetypes_pgf = ["dashed", "dashed", "solid", "solid"]
min_energy_col = "rgb,255: red,31; green,119; blue,180"    # blue: baseline scenarios
min_costs_color = "rgb,255: red,255; green,127; blue,14"   # orange: flexible operation scenarios
colors_pgf = [min_energy_col, min_costs_color, min_energy_col, min_costs_color]

plt_vec_acc_pgf = plot_KPI_acc_initialise_pgf()

# Capture requirement in the capture efficiency plot
push!(plt_vec_acc_pgf.cap_eff_acc, @pgf(PGFPlotsX.HLine({color = "black", style = "dotted", line_width = "1.5pt"}, 30.0)))
push!(plt_vec_acc_pgf.cap_eff_acc, "\\node[font=\\small] at (3, 30.8) {30\\% requirement};")

# Input bounds in the input plots
style_bound = @pgf {color = "black", style = "dashed", line_width = "1.5pt"}
push!(plt_vec_acc_pgf.F_plt, @pgf(PGFPlotsX.HLine(style_bound, 0.20)))
push!(plt_vec_acc_pgf.F_plt, @pgf(PGFPlotsX.HLine(style_bound, 0.80)))
push!(plt_vec_acc_pgf.Tld_plt, @pgf(PGFPlotsX.HLine(style_bound, 85.0)))
push!(plt_vec_acc_pgf.Tld_plt, @pgf(PGFPlotsX.HLine(style_bound, 95.0)))

for (i, d) in enumerate(test_data)
    plot_KPI_acc_pgf!(plt_vec_acc_pgf, d.df_test, d.t, d.p_el_time_grid, d.t_control, d.u_opt;
                      line = linetypes_pgf[i], color = colors_pgf[i], label = lbls[i])
end

fig13_names = (Q_acc = "fig13a_Q_acc", costs_pr_ton_acc = "fig13b_costs_pr_ton_acc", F_plt = "fig13c_F",
               cap_eff_acc = "fig13d_cap_eff_acc", SRD_acc = "fig13e_SRD_acc", Tld_plt = "fig13f_Td_sp")
for (key, name) in pairs(fig13_names)
    plt = plt_vec_acc_pgf[key]
    plt.options["width"] = "10.0cm"     # Size of the figures in the paper
    plt.options["height"] = "6.25cm"
    pgfsave(joinpath(FIG_DIR, "$(name).pdf"), plt)
end

## Figure 14: bar charts of the final KPIs (written as TikZ code, included with \input in the paper)
values_list = [kpi_df.Q_mean, kpi_df.costs_per_ton_CO2, kpi_df.cap_eff_final, kpi_df.SRD_final]
titles = ["", "", "", ""]
ylbls = ["{Reboiler duty, \$Q\$ [kW]}", "{Specific costs [\\texteuro/ton CO\$_2\$]}", "{Capture efficiency, \$\\eta\$ [\\%]}", "{SRD [MJ/kg CO\$_2\$]}"]
ymins = [minimum(values_list[i]) * 0.8 for i in 1:4]
ymins[2] = 0.0
ymaxs = [maximum(values_list[i]) for i in 1:4]
ymaxs_plt = ymaxs .+ [0.1, 0.15, 0.4, 0.1] .* (ymaxs .- ymins)
hlines = [nothing, nothing, (value = 30.0, label = "30\\% requirement"), nothing]

# Reduction of the specific costs by flexible operation in closed loop (#4 vs. #3)
cost_reduction = 100 * (1 - kpi_df.costs_per_ton_CO2[4] / kpi_df.costs_per_ton_CO2[3])
vertical_arrows = [nothing,
                   (x_idx = 4, y_start = values_list[2][3], y_end = values_list[2][4], label = "$(round(Int, cost_reduction))\\% \\\\ reduction"),
                   nothing, nothing]

fig14_dir = joinpath(FIG_DIR, "fig14_tex")
mkpath(fig14_dir)
kpi_grid_chart(values_list, titles, ymins, ymaxs_plt, ylbls, hlines, vertical_arrows, fig14_dir)

## Figure 12: energy price profiles
s = 0.7
plt = plot([time_pel[1], tf], [80.0, 80.0],
           label = "Baseline",
           lw = 2.0,
           xlabel = "Time [h]",
           ylabel = "Energy price signal,   " * L"p" * ", [EUR/MWh]",
           seriestype = :steppost,
           size = (600, 400) .* s,
           color = colors_palette.blue,
           legend_column_width = 100mm,
           legendfontsize = 9)
plot!(plt, [time_pel..., tf], [p_el; p_el[end]],
      label = "Flexible operation",
      lw = 2.0,
      seriestype = :steppost,
      color = colors_palette.orange)
savefig(plt, joinpath(FIG_DIR, "fig12_energy_price.pdf"))

## Final KPIs (Section 7.4)
CSV.write(joinpath(RESULTS_DIR, "empc_kpis.csv"), kpi_df)
println("\nFinal outputs of the four control tests:")
for row in eachrow(kpi_df)
    @printf("  #%d %-28s  capture eff. = %5.1f %%,  SRD = %4.2f MJ/kg CO2,  mean Q = %5.2f kW,  specific costs = %6.2f EUR/ton CO2\n",
            row.test, row.scenario, row.cap_eff_final, row.SRD_final, row.Q_mean, row.costs_per_ton_CO2)
end
@printf("  Reduction of specific costs, #4 vs. #3: %.1f %%\n", cost_reduction)

## OCP solve times in the closed-loop tests (Section 7.3)
solve_times = DataFrame(test = Int[], time_current = String[], solve_time_s = Float64[], termination_status = String[])
for test in eachrow(tests)
    stats_file = joinpath(tests_dir, test.folder, "solver_stats.json")
    isfile(stats_file) || continue
    for entry in JSON3.read(read(stats_file, String))
        push!(solve_times, (test.test, entry["timestamp_current"], entry["solve_time_s"], entry["termination_status"]))
    end
end
CSV.write(joinpath(RESULTS_DIR, "ocp_solve_times.csv"), solve_times)
@printf("\nOCP solve times, closed-loop tests #3 and #4 (%d control intervals): mean %.2f s, min %.2f s, max %.2f s\n",
        nrow(solve_times), mean(solve_times.solve_time_s), minimum(solve_times.solve_time_s), maximum(solve_times.solve_time_s))

println("\nSaved figures/fig12_energy_price.pdf, fig13*.pdf, fig14_tex/kpi_*.tex and results/empc_kpis.csv, results/ocp_solve_times.csv")
