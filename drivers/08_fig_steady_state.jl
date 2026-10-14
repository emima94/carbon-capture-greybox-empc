# ------------------------------------------------------------------------------------------
# Figure 8: Steady-state capture efficiency, reboiler duty and SRD (with capture efficiency
#           isolines) in the (F, Td_sp) input space
# Figure 9: SRD vs. solvent flow rate F at different capture efficiencies (U-curves)
#
# Uses the in-sample parameter estimate (all 10 series) with lambda_reg = 0.1.
#
# Output: figures/fig08a_cap_eff_contour.pdf, figures/fig08b_Q_contour.pdf,
#         figures/fig08c_SRD_contour.pdf, figures/fig09_SRD_u_curve.pdf,
#         results/srd_optimum.csv (minimum SRD and optimal F for each capture efficiency)
# Runtime: a few minutes (10,000 steady states on a 100 x 100 grid)
# ------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
set_plot_defaults(fontfamily = "times", titlefontsize = 9, guidefontsize = 9)

datasets, _ = load_pilot_plant_data()
est = only(filter(e -> e.sample_type == "in-sample", load_parameter_estimates(LOG10_LAMBDA_CHOSEN)))
par = est.p
model = build_dti_model(est.lambda_reg)
ffun_bar, hfun, p_fixed = model.ffun_bar, model.hfun, model.p_fixed

# Input and disturbance ranges of the data; the disturbances are fixed at the middle of their range
U_min, U_max, D_min, D_max = get_input_and_disturbance_bounds(datasets)
D_sim = (D_max + D_min) / 2

## Steady states in a (F, Td_sp) grid
N_grid = 100
F_range = range(U_min[1], U_max[1], length = N_grid)
TlD_SP_range = range(U_min[3], U_max[3], length = N_grid)

cap_eff_mat = zeros(N_grid, N_grid)
SRD_mat = zeros(N_grid, N_grid)
Q_mat = zeros(N_grid, N_grid)
ca_ss_mat = zeros(N_grid, N_grid)
cd_ss_mat = zeros(N_grid, N_grid)

println("Computing steady states on a $(N_grid) x $(N_grid) grid...")
for (i, F) in enumerate(F_range)
    for (j, TlD_SP) in enumerate(TlD_SP_range)
        U_ij = [F; NaN; TlD_SP]

        # Simulate 3 h (explicit Euler) to get a good initial guess, then solve 0 = f(x, u, d)
        dt = 1/60
        Nx = Int(3 / dt) + 1
        x_sim = zeros(3, Nx)
        x_sim[:, 1] = X0_GUESS
        for k in 1:Nx-1
            x_sim[:, k+1] = x_sim[:, k] .+ ffun_bar(x_sim[:, k], U_ij, D_sim, par, p_fixed) * dt
        end
        x_ss = nlsolve(x -> ffun_bar(x, U_ij, D_sim, par, p_fixed), x_sim[:, end]).zero

        ca_ss_mat[i, j] = x_ss[1]
        cd_ss_mat[i, j] = x_ss[2]
        y_ss_named = named_output(hfun(x_ss, U_ij, D_sim, par, p_fixed))
        cap_eff_mat[i, j] = y_ss_named.cap_eff[1]   # Capture efficiency [%]
        Q_mat[i, j] = y_ss_named.Q[1]               # Reboiler duty [kW]
        SRD_mat[i, j] = y_ss_named.SRD[1]           # Specific reboiler duty [MJ/kg CO2]
    end
end

## Figure 8: contour plots
y_lbl = latexstring("\\textrm{Desorber\\ temperature\\ SP,}\\ T_{d}^{sp}\\textrm{,\\ [°C]}")
x_lbl = latexstring("\\textrm{Solvent\\ flow\\ rate,}\\ F\\textrm{,\\ [m}^3\\textrm{/h]}")
right_margin = 6mm
plt_size = (600, 500) .* 0.6
cbar_title_cap_eff = latexstring("\\textrm{Capture\\ efficiency,\\ } \\eta \\textrm{\\ [\\%]}")
cbar_title_SRD = latexstring("\\textrm{SRD\\ [MJ/kg\\ CO}_2\\textrm{]}")
cbar_title_Q = latexstring("\\textrm{Reboiler\\ duty,\\ } Q \\textrm{\\ [kW]}")

# Colorbar titles are added as rotated annotations to the right of the plot
function contour_plot(mat, clims, cbar_title, x_rel_cbar_title)
    plt = contour(F_range, TlD_SP_range, mat', xlabel = x_lbl, ylabel = y_lbl, fill = true,
                  clims = clims, size = plt_size, right_margin = right_margin)
    annotate!(plt,
              x_rel_cbar_title * (F_range[end] - F_range[1]) + F_range[1],
              0.5 * (TlD_SP_range[end] - TlD_SP_range[1]) + TlD_SP_range[1],
              text(cbar_title, font(10), rotation = 90))
    return plt
end

plt_cap_eff_mat = contour_plot(cap_eff_mat, (10, 50), cbar_title_cap_eff, 1.22)
plt_Q_mat = contour_plot(Q_mat, (0.0, 25.0), cbar_title_Q, 1.22)
plt_SRD_mat = contour_plot(SRD_mat, (4.5, 8.5), cbar_title_SRD, 1.24)

# Capture efficiency isolines on the SRD contour plot
eta_min_list = [20, 25, 30, 35, 40]
colors = palette(:tab10)[1:length(eta_min_list)]

# Grid indices of the isolines: for each F, the first Td_sp where cap_eff >= eta_min.
# Used to mark which parts of the U-curves are inside the Td_sp range of the data.
iso_lines_idxs = Dict{Float64, Tuple{Vector{Int}, Vector{Int}}}()
for eta_min in eta_min_list
    idx_y_vec = findfirst.(Ref(>=(eta_min)), eachrow(cap_eff_mat))
    valid_idx = [i for (i, idx_y) in enumerate(idx_y_vec) if idx_y !== nothing &&
                 !(idx_y == 1 && all(cap_eff_mat[i, :] .>= eta_min))]
    iso_lines_idxs[eta_min] = (valid_idx, [idx_y_vec[i] for i in valid_idx])
end

# Smooth isolines by linear interpolation between grid points, extrapolated to cover the full F range
for (eta_min, color) in zip(eta_min_list, colors)
    x_line = Float64[]
    y_line = Float64[]
    for (i, row) in enumerate(eachrow(cap_eff_mat))
        idx = findfirst(>=(eta_min), row)
        isnothing(idx) && continue
        idx == 1 && all(row .>= eta_min) && continue
        if idx > 1
            v0 = row[idx - 1]
            v1 = row[idx]
            t = (eta_min - v0) / (v1 - v0)
            y_interp = TlD_SP_range[idx - 1] + t * (TlD_SP_range[idx] - TlD_SP_range[idx - 1])
        else
            y_interp = TlD_SP_range[idx]
        end
        push!(x_line, F_range[i])
        push!(y_line, y_interp)
    end

    if length(x_line) >= 2
        if x_line[1] > F_range[1]
            slope = (y_line[2] - y_line[1]) / (x_line[2] - x_line[1])
            pushfirst!(y_line, y_line[1] + slope * (F_range[1] - x_line[1]))
            pushfirst!(x_line, F_range[1])
        end
        if x_line[end] < F_range[end]
            slope = (y_line[end] - y_line[end-1]) / (x_line[end] - x_line[end-1])
            push!(y_line, y_line[end] + slope * (F_range[end] - x_line[end]))
            push!(x_line, F_range[end])
        end
    end
    isempty(x_line) && continue

    plot!(plt_SRD_mat, x_line, y_line,
          color = :black, linewidth = 5, linestyle = :solid, label = "", ylims = (85.0, 95.0), xlims = (F_range[1], F_range[end]))
    plot!(plt_SRD_mat, x_line, y_line,
          color = color, linewidth = 3, linestyle = :solid, label = latexstring("\\eta = $(eta_min) \\%"),
          ylims = (85.0, 95.0), xlims = (F_range[1], F_range[end]))
end
plot!(plt_SRD_mat, legend = :topright)

savefig(plt_cap_eff_mat, joinpath(FIG_DIR, "fig08a_cap_eff_contour.pdf"))
savefig(plt_Q_mat, joinpath(FIG_DIR, "fig08b_Q_contour.pdf"))
savefig(plt_SRD_mat, joinpath(FIG_DIR, "fig08c_SRD_contour.pdf"))

## Figure 9: SRD vs. F along the capture efficiency isolines
# For each F and capture efficiency target, solve for the steady state and the Td_sp that give
# the target capture efficiency: z = [ca, cd, Tld_HX, Td_sp] (gfun in src/util_diffeq_sys_id.jl)
SRD = [zeros(N_grid) for _ in eachindex(eta_min_list)]
println("Computing SRD U-curves...")
for (i, eta_min) in enumerate(eta_min_list)
    for (j, F) in enumerate(F_range)
        k = Int(round(N_grid / 2))  # Middle of the Td_sp range as initial guess
        z_init = [ca_ss_mat[j, k], cd_ss_mat[j, k], TlD_SP_range[k], TlD_SP_range[k]]
        z_opt = nlsolve(z -> gfun(z, F, D_sim, par, p_fixed, eta_min, ffun_bar, hfun), z_init).zero
        y_opt_named = named_output(hfun(z_opt, [F; NaN; z_opt[4]], D_sim, par, p_fixed))
        SRD[i][j] = y_opt_named.SRD[1]
    end
end

plt_SRD_u_curve = plot(xlabel = latexstring("\\textrm{Solvent\\ flow\\ rate,}\\ F\\textrm{,\\ [m}^3\\textrm{/h]}"),
                       ylabel = latexstring("\\textrm{SRD\\ [MJ/kg\\ CO}_2\\textrm{]}"),
                       legend = (0.82, 0.34),
                       ylims = (5.5, 7.5),
                       size = (600, 400) .* 0.8)

srd_optimum = DataFrame(cap_eff = Float64[], F_opt = Float64[], SRD_min = Float64[])
for (i, eta_min) in enumerate(eta_min_list)
    # Solid: Td_sp inside the range of the data (85-95 °C). Dotted: outside (extrapolation).
    x_idxs_in_range, _ = iso_lines_idxs[eta_min]
    x_idxs_out_of_range = setdiff(1:length(F_range), x_idxs_in_range)
    plot!(plt_SRD_u_curve, F_range[x_idxs_in_range], SRD[i][x_idxs_in_range], label = latexstring("\\eta = $(eta_min) \\%"),
          lw = 2.0, seriestype = :line, color = colors[i])
    x_idxs_left = [x_idxs_out_of_range[x_idxs_out_of_range .< minimum(x_idxs_in_range)]; x_idxs_in_range[1]]
    plot!(plt_SRD_u_curve, F_range[x_idxs_left], SRD[i][x_idxs_left], label = "", lw = 2.0, seriestype = :line, linestyle = :dot, color = colors[i])
    x_idxs_right = [x_idxs_in_range[end]; x_idxs_out_of_range[x_idxs_out_of_range .> maximum(x_idxs_in_range)]]
    plot!(plt_SRD_u_curve, F_range[x_idxs_right], SRD[i][x_idxs_right], label = "", lw = 2.0, seriestype = :line, linestyle = :dot, color = colors[i])

    # Minimum of the U-curve within the range of the data
    j_opt = x_idxs_in_range[argmin(SRD[i][x_idxs_in_range])]
    push!(srd_optimum, (eta_min, F_range[j_opt], SRD[i][j_opt]))
end
savefig(plt_SRD_u_curve, joinpath(FIG_DIR, "fig09_SRD_u_curve.pdf"))
CSV.write(joinpath(RESULTS_DIR, "srd_optimum.csv"), srd_optimum)

println("\nMinimum SRD along each capture efficiency isoline (Section 6.2):")
for row in eachrow(srd_optimum)
    @printf("  eta = %2.0f %%:  F_opt = %.2f m3/h,  SRD_min = %.2f MJ/kg CO2\n", row.cap_eff, row.F_opt, row.SRD_min)
end
println("\nSaved figures/fig08a_cap_eff_contour.pdf, fig08b_Q_contour.pdf, fig08c_SRD_contour.pdf, fig09_SRD_u_curve.pdf")
