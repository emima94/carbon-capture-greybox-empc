
function plot_ekf_predictions_offline!(plt_vec, u, Mgina_final, time_control, time_final, out_pred, settings)

    # Extract results from out_pred
    time_post = out_pred.time_post
    x_post = out_pred.x_post
    y_out_post_mat = out_pred.y_out_post_mat
    time_pred = out_pred.time_pred
    idx_pred = time_pred .<= time_post[end] # Only consider predictions within the time range of the data
    time_pred = time_pred[idx_pred]
    x_pred = out_pred.x_pred[:, idx_pred]
    y_pred = out_pred.y_pred[:, idx_pred]

    time_step_ahead = settings.time_step_ahead
    time_start = settings.time_start
    u_prev = settings.u_prev
    N = settings.N
    #dt_control_interval = settings.dt_control_interval
    #time_final = time_start + Second(Int(round((N) * dt_control_interval * 3600)))
    eta_min = settings.eta_min
    #time_control = collect((0:N-1) * dt_control_interval)
    
    
    time_stamps_control = Millisecond.(round.(time_control*1000*3600)) .+ time_start
    #time_stamps_control_plt = [time_stamps_control; time_stamps_control[end]]
    time_stamps_control_plt = [time_stamps_control; time_final]

    # Named vars
    z_out_post = named_output(y_out_post_mat)
    z_pred = named_output(y_pred)

    #
    #Mga_sim = cumsum(z_sim.ma .* diff([0; time_sim]) .* Hour(1)) # Cumulative sum of capture rate * time step (converted to hours)
    
    # Plot 30 % requirement as horizontal line
    Mga_req = eta_min/100 * Mgina_final
    hline!(plt_vec.acc_CO2_plt, [Mga_req], label = "$eta_min% req.", lw = 1.0, ls = :dash, color = :red)
    # Plot vertical line indicating finish of test period
    #vline!(plt_vec.acc_CO2_plt, [time_f], label = "", lw = 1.0, ls = :dash, color = :black)

    # Capture efficiency plot
    plot!(plt_vec.cap_eff_plt, time_post, z_out_post.cap_eff, label = "EKF post.", lw = lw)
    plot!(plt_vec.cap_eff_plt, time_pred, z_pred.cap_eff, label = "$(time_step_ahead) h ahead pred", lw = lw, ls = :dash)
    
    # Specific reboiler duty
    plot!(plt_vec.srd_plt, time_post, z_out_post.SRD, label = "EKF post.", lw = lw)
    plot!(plt_vec.srd_plt, time_pred, z_pred.SRD, label = "$(time_step_ahead) h ahead pred", lw = lw, ls = :dash)

    # Co2 gas conc plot
    plot!(plt_vec.co2_conc_plt, time_post, z_out_post.yga, label = "EKF post.", lw = lw)
    plot!(plt_vec.co2_conc_plt, time_pred, z_pred.yga, label = "$(time_step_ahead) h ahead pred", lw = lw, ls = :dash)

    # CO2 capture rate
    plot!(plt_vec.mga_plt, time_post, z_out_post.ma, label = "EKF post.", lw = lw)
    plot!(plt_vec.mga_plt, time_pred, z_pred.ma, label = "$(time_step_ahead) h ahead pred", lw = lw, ls = :dash)

   
    # Solvent flow rate plot as stair plot

    plot!(plt_vec.F_plt, time_stamps_control_plt, [u_prev[1]; [u[i][1] for i in eachindex(u)]...], seriestype = :steppre, label = "Control input", lw = lw, ls = :dash)
     
    # Temperature desorber plot
    plot!(plt_vec.Tld_plt, time_stamps_control_plt, [u_prev[2]; [u[i][2] for i in eachindex(u)]...], seriestype = :steppre, label = "Control input", lw = lw, ls = :dash)
    
    # Reboiler duty plot
    plot!(plt_vec.Q_plt, time_post, z_out_post.Q, label = "EKF post.", lw = lw)
    plot!(plt_vec.Q_plt, time_pred, z_pred.Q, label = "$(time_step_ahead) h ahead pred", lw = lw, ls = :dash)

   
    # States plot
    plot!(plt_vec.states_plt, time_post, x_post[1:2,:]', label = "EKF post.", lw = lw)
    plot!(plt_vec.states_plt, time_pred, x_pred[1:2,:]', label = "$(time_step_ahead) h ahead pred", lw = lw, ls = :dash)

    # Set xticks format for all plots
    min_time = minimum([time_post[1], time_pred[1]])
    max_time = maximum([time_post[end], time_pred[end]])
    for plt in values(plt_vec)
        plot!(plt, xticks = get_x_ticks_fmt(time_start, time_final))
    end

    return plt_vec

end

function plot_KPI_acc!(plt_vec, df, t, p_el; line=:solid, color =:blue, label = "Test")

    # Constants
    kW2MW = 1e-3
    kWh2MJ = 3.6

    dt = diff(t)

    Q_acc = [0; cumsum(df.Q .* dt)]
    costs_acc = [0; cumsum(p_el .* df.Q * kW2MW .* dt)]
    cap_eff_acc = cumsum(df.mga_CO2) ./ cumsum(df.mgina_CO2) * 100
    SRD_acc = cumsum(df.Q) * kWh2MJ ./ cumsum(df.mga_CO2)


    plot!(plt_vec.Q_acc, t, Q_acc, label=label, line = line, color = color)
    plot!(plt_vec.costs_acc, t, costs_acc, label=label, line = line, color = color)
    plot!(plt_vec.cap_eff_acc, t[1:end-1], cap_eff_acc, label=label, line = line, color = color)
    plot!(plt_vec.SRD_acc, t[1:end-1], SRD_acc, label=label, line = line, color = color)

    return plt_vec

end

function plot_KPI_acc_initialise()

    plt_Q_acc = plot(title="Acc. Reboiler Duty", xlabel="Time [h]", ylabel="Reboiler duty [kWh]", legend=:topright)
    plt_costs_acc = plot(title="Acc. Costs", xlabel="Time [h]", ylabel="Costs [EUR]", legend=:topright)
    plt_cap_eff_acc = plot(title="Mean Capture Efficiency", xlabel="Time [h]", ylabel="Accumulated Capture Efficiency [%]", legend=:topright)
    plt_SRD_acc = plot(title="Mean Specific Energy", xlabel="Time [h]", ylabel="Accumulated Specific Energy [MJ/kg CO2]", legend=:topright)

    return (Q_acc = plt_Q_acc, costs_acc = plt_costs_acc, cap_eff_acc = plt_cap_eff_acc, SRD_acc = plt_SRD_acc)

end


function plot_KPI_acc_initialise_pgf() 

    title_y_pos = 1.08
    xmin = 0
    xmax = 6
    ymin = 0
    title_font_size = "\\normalsize"
    lbl_font_size = "\\normalsize"

      kw_Q = @pgf {
        #title  = "Accumulated Reboiler Duty [kWh]",
        #title_style =  {at = "{(0.5, $title_y_pos)}", anchor = "north", font = title_font_size},
        xlabel = "Time [h]",
        ylabel = raw"Reboiler duty, $Q$ [kW]",
        xlabel_style = {font = lbl_font_size},
        ylabel_style = {font = lbl_font_size},
        legend_pos = "north west",
            legend_style = {"nodes={anchor=west}"},
            grid = "both",
        xmin = xmin, xmax = xmax,
        ymin = 9, ymax = 20,
    }

    kw_costs = @pgf {
        #title  = "Accumulated Costs [EUR]",
        #title = "",
        #title_style = {at = "{(0.5, $title_y_pos)}", anchor = "north", font = title_font_size},
        xlabel = "Time [h]",
        ylabel = "Costs [EUR]",
        xlabel_style = {font = lbl_font_size},
        ylabel_style = {font = lbl_font_size},
        legend_pos = "north west",
        legend_style = {"nodes={anchor=west}"},
        grid = "both",
        xmin = xmin, xmax = xmax,
        ymin = ymin,
    }

    kw_costs_pr_ton = @pgf {
        #title  = raw"Mean Specific Costs [EUR/ton CO$_2$]",
        #title = "",
        #title_style = {at = "{(0.5, $title_y_pos)}", anchor = "north", font = title_font_size},
        xlabel = "Time [h]",
        ylabel = raw"Specific costs [\texteuro/ton CO$_2$]",
        xlabel_style = {font = lbl_font_size},
        ylabel_style = {font = lbl_font_size},
        #legend_pos = "north",
        #legend_style = {"nodes={anchor=west}"},
        legend_style = {at= "(0.5, 1.1)", anchor = "north", "nodes={anchor=west}"},
        grid = "both",
        xmin = xmin, xmax = xmax,
        ymax = 200,
        #ymin = ymin,
    }

    kw_cap = @pgf {
        #title  = "Mean Capture Efficiency [\\%]",
        #title = "",
        #title_style = {at = "{(0.5, $title_y_pos)}", anchor = "north", font = title_font_size},
        xlabel = "Time [h]",
        ylabel = raw"Capture efficiency, $\eta$ [\%]",
        xlabel_style = {font = lbl_font_size},
        ylabel_style = {font = lbl_font_size},
        legend_pos = "north west",
        legend_style = {"nodes={anchor=west}"},
        grid = "both",
        xmin = xmin, xmax = xmax,
    }

    kw_srd = @pgf {
        #title  = raw"Mean Specific Reboiler Duty [MJ/kg CO$_2$]",
        #title = "",
        #title_style = {at = "{(0.5, $title_y_pos)}", anchor = "north", font = title_font_size},
        xlabel = "Time [h]",
        ylabel = raw"SRD [MJ/kg CO$_2$]",
        xlabel_style = {font = lbl_font_size},
        ylabel_style = {font = lbl_font_size},
        legend_pos = "north west",
        legend_style = {"nodes={anchor=west}"},
        grid = "both",
        xmin = xmin, xmax = xmax,
    }

    kw_F = @pgf {
        xlabel = "Time [h]",
        ylabel = raw"Solvent flow rate, $F$ [m$^3$/h]",
        xlabel_style = {font = lbl_font_size},
        ylabel_style = {font = lbl_font_size},
        legend_pos = "north west",
        legend_style = {"nodes={anchor=west}"},
        grid = "both",
        xmin = xmin, xmax = xmax,
        ymin = 0.1, ymax = 0.9,
    }

    kw_Tld = @pgf {
        xlabel = "Time [h]",
        ylabel = raw"Desorber temperature, $T_{d}^{sp}$ [°C]",
        xlabel_style = {font = lbl_font_size},
        ylabel_style = {font = lbl_font_size},
        legend_pos = "north west",
        legend_style = {"nodes={anchor=west}"},
        grid = "both",
        xmin = xmin, xmax = xmax,
        ymin = 84, ymax = 96
    }

    return (
        Q_acc       = @pgf(PGFPlotsX.Axis(kw_Q)),
        costs_acc   = @pgf(PGFPlotsX.Axis(kw_costs)),
        costs_pr_ton_acc = @pgf(PGFPlotsX.Axis(kw_costs_pr_ton)),
        cap_eff_acc = @pgf(PGFPlotsX.Axis(kw_cap)),
        SRD_acc     = @pgf(PGFPlotsX.Axis(kw_srd)),
        F_plt       = @pgf(PGFPlotsX.Axis(kw_F)),
        Tld_plt     = @pgf(PGFPlotsX.Axis(kw_Tld)),
    )
end

function plot_KPI_acc_pgf!(plt_vec, df, t, p_el, t_control, u_opt;
                       line  = "solid",
                       color = "blue",
                       label = "Test")

    kW2MW  = 1e-3
    kWh2MJ = 3.6

    u_opt_mat = hcat(u_opt...) # Convert vector of control input vectors to a matrix for easier indexing

    dt = diff(t)

    #Q_acc       = [0; cumsum(df.Q .* dt)] 
    Q_mean      = cumsum(df.Q .* dt) ./ t[1:end-1] # Mean reboiler duty up to each time point 
    costs_acc   = [0; cumsum(p_el .* df.Q * kW2MW .* dt)]
    cost_pr_ton_acc = cumsum(p_el .* df.Q * kW2MW) ./ (cumsum(df.mga_CO2) * 1e-3) 
    cap_eff_acc = cumsum(df.mga_CO2)  ./ cumsum(df.mgina_CO2) * 100
    SRD_acc     = cumsum(df.Q) * kWh2MJ ./ cumsum(df.mga_CO2)

    style = @pgf {
        color      = color,
        line_width = "1.5pt",
        style      = line,
        mark       = "none",
    }

    style_stair = @pgf {
        const_plot,
        color      = color,
        line_width = "1.5pt",
        style      = line,
        mark       = "none",
    }

    add!(ax, xs, ys, lbl) = push!(ax,
        @pgf(PlotInc(style, Coordinates(xs, ys))),
        LegendEntry(lbl),
    )

    add_no_lgd!(ax, xs, ys, lbl) = push!(ax,
        @pgf(PlotInc(style, Coordinates(xs, ys))),
    )

    add_stair_no_lgd!(ax, xs, ys, lbl) = push!(ax,
        @pgf(PlotInc(style_stair, Coordinates(xs, ys))),
    )


    add_stair_no_lgd!(plt_vec.F_plt, t_control, vcat(u_opt_mat[1,:], u_opt_mat[1,end]), label) # Add control input trajectory to F plot, without legend entry
    add_stair_no_lgd!(plt_vec.Tld_plt, t_control, vcat(u_opt_mat[2,:], u_opt_mat[2,end]), label) # Add control input trajectory to Tld plot, without legend entry
    add_no_lgd!(plt_vec.Q_acc,       t[1:end-1],          Q_mean,       label)
    add!(plt_vec.costs_acc,   t,          costs_acc,   label)
    add!(plt_vec.costs_pr_ton_acc, t[1:end-1], cost_pr_ton_acc, label)
    add_no_lgd!(plt_vec.cap_eff_acc, t[1:end-1], cap_eff_acc, label)
    add_no_lgd!(plt_vec.SRD_acc,     t[1:end-1], SRD_acc,     label)

    return plt_vec
end






function kpi_bar_plot_content(values, title, xtick_pos, bar_width, ymin, ymax, ylbl, hline = nothing, vertical_arrow = nothing)

    # x- and y_pos group annotations
    x_pos_annotation = [(xtick_pos[1] + xtick_pos[2]) / 2, (xtick_pos[3] + xtick_pos[4]) / 2]
    y_pos_annotation = ymin - (ymax - ymin) * 0.30

    if hline !== nothing
        hline_tex = """
        % Horizontal line for requirement
        \\draw[dotted, black, line width=1pt] (axis cs:$(xtick_pos[1]-bar_width),$(hline.value)) -- (axis cs:$(xtick_pos[4]+bar_width),$(hline.value));
        """
        hline_annotation_tex = """
        % Annotation for horizontal line
        \\node[above, font=\\footnotesize] at (axis cs:$((xtick_pos[4]+xtick_pos[1])/2),$(hline.value)) {$(hline.label)};
        """
        #hline_tex *= hline_annotation_tex
    else
        hline_tex = ""
        hline_annotation_tex = ""
    end

    if vertical_arrow !== nothing
        vertical_arrow_tex = """
        % Vertical arrow for reduction
        \\draw[|->, thick] (axis cs:$(xtick_pos[vertical_arrow.x_idx]-bar_width/2),$(vertical_arrow.y_start)) -- (axis cs:$(xtick_pos[vertical_arrow.x_idx]-bar_width/2),$(vertical_arrow.y_end)) node[midway, anchor=west, font=\\footnotesize, align=left] {$(vertical_arrow.label)};
        """
    else
        vertical_arrow_tex = ""
    end

    # Color definitions
    color_defs = """
    \\definecolor{minEnergyColor}{HTML}{1f77b4}
    \\definecolor{minCostsColor}{HTML}{ff7f0e}
    """

    return """
    $color_defs

    % Open-loop: Min Energy
    \\fill[draw=minEnergyColor, pattern=north east lines, pattern color=minEnergyColor] (axis cs:$(xtick_pos[1] - bar_width/2),$ymin) rectangle (axis cs:$(xtick_pos[1] + bar_width/2),$(values[1]));

    % Open-loop: Min Costs
    \\fill[draw=minCostsColor, pattern=north east lines, pattern color=minCostsColor] (axis cs:$(xtick_pos[2] - bar_width/2),$ymin) rectangle (axis cs:$(xtick_pos[2] + bar_width/2),$(values[2]));

    % Closed-loop: Min Energy
    \\fill[minEnergyColor!50, draw=minEnergyColor] (axis cs:$(xtick_pos[3] - bar_width/2),$ymin) rectangle (axis cs:$(xtick_pos[3] + bar_width/2),$(values[3]));

    % Closed-loop: Min Costs
    \\fill[minCostsColor!50, draw=minCostsColor] (axis cs:$(xtick_pos[4] - bar_width/2),$ymin) rectangle (axis cs:$(xtick_pos[4] + bar_width/2),$(values[4]));

    % Optional horizontal line + annotation for requirement
    $hline_tex
    $hline_annotation_tex

    % Optional vertical arrow + annotation for reduction
    $vertical_arrow_tex

    % Group annotations
    \\node[below, font=\\footnotesize] at (axis cs:$(x_pos_annotation[1]),$y_pos_annotation) {Open-loop};
    \\node[below, font=\\footnotesize] at (axis cs:$(x_pos_annotation[2]),$y_pos_annotation) {Closed-loop};
    """
end


function kpi_grid_chart(values_list, titles, ymins, ymaxs, ylbls, hlines, vertical_arrows,
    output_dir = "."
)
    w = 7
    h = 5
    scale = 1.1
    bar_width = 0.3
    sub_group_sep = 0.5
    group_sep = 0.7
    start_pos = 1.0
    xtick_pos = [start_pos, 
                    start_pos + sub_group_sep, 
                    start_pos + sub_group_sep + group_sep, 
                    start_pos + sub_group_sep + group_sep + sub_group_sep]

    for i in 1:4
        plt = kpi_bar_plot_content(values_list[i], titles[i], 
                                        xtick_pos, 
                                        bar_width, 
                                        ymins[i], 
                                        ymaxs[i], 
                                        ylbls[i], 
                                        hlines[i],
                                        vertical_arrows[i])

        tex = """
        \\begin{tikzpicture}[scale=1.0]
        \\begin{axis}[
            title={$(titles[i])},
            title style={yshift=-5mm},
            ylabel=$(ylbls[i]),
            ylabel style={yshift=-1mm},
            xmin=$(xtick_pos[1] - bar_width), xmax=$(xtick_pos[4] + bar_width),
            ymin=$(ymins[i]), ymax=$(ymaxs[i]),
            xtick={$(join(xtick_pos, ", "))},
            xticklabels={
                {\\#1 \\\\ Baseline \\\\~},
                {\\#2 \\\\ Flexible \\\\ operation},
                {\\#3 \\\\ Baseline\\\\~},
                {\\#4 \\\\ Flexible \\\\ operation}
            },
            tick label style={align=center, font=\\footnotesize},
            axis x line=bottom,
            axis y line=left,
            axis on top=true,
            ymajorgrids=true,
            grid style={dashed, gray!40},
            width=$(w * scale)cm,
            height=$(h * scale)cm,
            clip=false,
        ]

        $plt

        \\end{axis}
        \\end{tikzpicture}
        """

        filename = joinpath(output_dir, "kpi_$(i).tex")
        write(filename, tex)
        println("Wrote $filename")
    end
end

# Old tick lbls:
        #  {Min.\\\\Energy},
        #         {Min.\\\\Costs},
        #         {Min.\\\\Energy},
        #         {Min.\\\\Costs}