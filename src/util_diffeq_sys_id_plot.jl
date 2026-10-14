

function initialise_plots(ds, ms, plant; bounds = nothing, w_Ta = true, seriestype = :scatter)

    lw = 2.0 # linewidth

    R = 8.314 # J/(mol K)
    T = 313.15 # K, assumed constant temperature in absorber for calculating ygina  
    P = 101325 # Pa, assumed constant pressure in absorber
    cg_tot = P/(R*T)
    cgina = ds.dist[1,:]
    ygina = cgina / cg_tot * 100

    if plant == "DTI"
        #ylims_x = (0,5.0)
        ylims_x = (2,3)
        ylims_yga = (0, 18)
        if w_Ta
            ylims_TlD_HX = (25, 100)
        else
             ylims_TlD_HX = (80,100)
        end
#        ylims_TlD_HX = (25, 100)
        ylims_Preb = (-2, 30)
        ylims_Fco = (0.0, 1.0)
        ylims_cgina = (4.0, 7.0)
        ylims_Fgina = (0.0, 150.0)
        ylims_N = (0, 300)
        ylims_SRD = (1.0, 10.0)
        ylims_cap_eff = (0.0, 100.0)
    elseif plant == "Tiller"
        ylims_x = (-1, 5.0)
        ylims_yga = (0, 4.0)
        ylims_TlD_HX = (0, 0) # Not used for Tiller
        ylims_Preb = (20, 35)
        ylims_Fco = (0.0, 0.4)
        ylims_cgina = (4.0, 7.0)
        ylims_Fgina = (0.0, 200.0)
        ylims_N = (400, 900)
        ylims_SRD = (3.0, 10.0)
        ylims_cap_eff = (10.0, 100.0)
    else
        error("Unknown plant type: $(plant)")
    end

    ylims_d_adaptive = (-0.5, 0.5) # Y-limits for the adaptive disturbance term plot

    plt_x = plot(ylabel = "CO2 conc. liq. [kmol/m3]", ylims = ylims_x)

    plt_yga = plot(ylabel = "CO2 conc. gas [mol/m3]", ylims = ylims_yga)
    
    yga = ds.Y[1,:]
    if !isnan(yga[1])
        plot!(plt_yga, ds.t, yga, label = "yga, data", seriestype = :scatter, ms = ms, color = :red, markerstrokecolor = :red)
        #plot!(plt_yga, ds.t, ygina, label = "ygina", seriestype = :scatter, ms = ms, color = :red, markerstrokecolor = :red)
    else
        plot!(plt_yga, ds.t, ygina, label = "ygina", lw = lw, color = :green, linestyle = :solid)
    end

    plt_TlD_HX = plot(ylabel = "Temperature [C]", ylims = ylims_TlD_HX)

    if plant == "DTI"
       
        plot!(plt_TlD_HX, ds.t, ds.u[3,:], label = "TlD, SP", linestyle = :dash, color = :red)
        if w_Ta
             plot!(plt_TlD_HX, ds.t, ds.u[2,:], label = "TlA, SP", linestyle = :dash, color = :blue)
        end
        #plot!(plt_TlD_HX, ds.t, ds.u[2,:], label = "TlA, SP", linestyle = :dash, color = :blue)
        if !isnan(ds.Y[2,1])
            plot!(plt_TlD_HX, ds.t, ds.Y[2,:], label = "TlA, data", seriestype = :scatter, color = :blue, markerstrokecolor = :blue, ms = ms)
        end
        if !isnan(ds.Y[3,1])
            plot!(plt_TlD_HX, ds.t, ds.Y[3,:], label = "TlD, data", seriestype = :scatter, color = :red, markerstrokecolor = :red, ms = ms)
        end
    else
        #plt_TlD_HX = plot();
    end

    plt_Preb = plot(ylabel = "Reboiler duty [kW]", ylims = ylims_Preb)

    if plant == "DTI"
        Preb = ds.Y[4,:]
        if !isnan(Preb[1])
            plot!(plt_Preb, ds.t, Preb, label = "Data", seriestype = :scatter, color = :red, markerstrokecolor = :red, ms = ms)
        end
    elseif plant == "Tiller"
        Preb = ds.u[2,:] # Reboiler duty is an input in Tiller
        plot!(plt_Preb, ds.t, Preb, label = "Reboiler duty, SP", linestyle = :dash, color = :red)
    end
    
    plt_Fco = plot(ds.t, ds.u[1,:], ylabel = "Flow rate [m3/h]", ylims = ylims_Fco, label ="")
    plt_cgina = plot(ds.t, ds.dist[1,:], ylabel = "Inlet CO2 conc. [mol/m3]", ylims = ylims_cgina, label ="")
    plt_Fgina = plot(ds.t, ds.dist[2,:], ylabel = "Gas flow rate [m3/h]", ylims = ylims_Fgina, label ="")
    plt_N = plot(label = "Na", ylabel = "Capture rate [mol/h]", ylims = ylims_N)
    
    plt_SRD = plot(label = "SRD", ylabel = "SRD [kg/m3]", ylims = ylims_SRD)
    plt_cap_eff = plot(label = "Cap. Eff.", ylabel = "Capacity Efficiency [%]", ylims = ylims_cap_eff)
    plt_d_adaptive = plot(ylabel = "Disturbance", ylims = ylims_d_adaptive)

    # Add bounds to plots if provided
    if !isnothing(bounds)
        U_min, U_max, D_min, D_max = bounds
        plot!(plt_Fco, ds.t, fill(U_min[1], length(ds.t)), label = "", linestyle = :dash, color = :black)
        plot!(plt_Fco, ds.t, fill(U_max[1], length(ds.t)), label = "", linestyle = :dash, color = :black)
        plot!(plt_cgina, ds.t, fill(D_min[1], length(ds.t)), label = "", linestyle = :dash, color = :black)
        plot!(plt_cgina, ds.t, fill(D_max[1], length(ds.t)), label = "", linestyle = :dash, color = :black)
        plot!(plt_Fgina, ds.t, fill(D_min[2], length(ds.t)), label = "", linestyle = :dash, color = :black)
        plot!(plt_Fgina, ds.t, fill(D_max[2], length(ds.t)), label = "", linestyle = :dash, color = :black)
        if plant == "DTI"
            plot!(plt_TlD_HX, ds.t, fill(U_min[2], length(ds.t)), label = "", linestyle = :dash, color = :black)
            plot!(plt_TlD_HX, ds.t, fill(U_max[2], length(ds.t)), label = "", linestyle = :dash, color = :black)
            plot!(plt_TlD_HX, ds.t, fill(U_min[3], length(ds.t)), label = "", linestyle = :dash, color = :black)
            plot!(plt_TlD_HX, ds.t, fill(U_max[3], length(ds.t)), label = "", linestyle = :dash, color = :black)
        else
            plot!(plt_Preb, ds.t, fill(U_min[2], length(ds.t)), label = "", linestyle = :dash, color = :black)
            plot!(plt_Preb, ds.t, fill(U_max[2], length(ds.t)), label = "", linestyle = :dash, color = :black)
        end
    end


    plts = (plt_yga = plt_yga, plt_TlD_HX = plt_TlD_HX, plt_Preb = plt_Preb, plt_Fco = plt_Fco, 
        plt_cgina = plt_cgina, plt_Fgina = plt_Fgina, plt_N = plt_N, plt_x = plt_x, plt_SRD = plt_SRD, plt_cap_eff = plt_cap_eff, plt_d_adaptive = plt_d_adaptive)

end


function initialise_plots_paper(ds, plant)

    ms = 1.50

    if plant == "DTI"
        ylims_yga = (-1, 15)
        ylims_Q = (0.0, 30)
        ylims_d_adaptive = (-0.25, 0.25)
    elseif plant == "Tiller"
        ylims_yga = (0, 4.0)
        ylims_Q = (20, 35)
        ylims_d_adaptive = (-0.25, 0.25)
    else
        error("Unknown plant type: $(plant)")
    end

    #yga_ylbl = "CO" * L"_2" * " gas conc.\n " * L"\hat{y}_{k+j|k}" * " [vol%]"
    yga_ylbl = latexstring("\\textrm{CO}_2 \\textrm{\\ gas\\ conc.,}") * "\n" * latexstring("y_g\\ \\textrm{[vol\\%]}")
    plt_yga = plot(ylabel = yga_ylbl, ylims = ylims_yga)
    
    yga = ds.Y[1,:]
    plot!(plt_yga, ds.t, yga, label = "Data", seriestype = :scatter, ms = ms, color = :red, markerstrokewidth = 0.0)
    
    Q_ylbl = latexstring("\\textrm{Reboiler\\ duty,}") * "\n" * latexstring("Q\\ \\textrm{[kW]}")
    plt_Q = plot(ylabel = Q_ylbl, ylims = ylims_Q)

    if plant == "DTI"
        Q = ds.Y[4,:]
        plot!(plt_Q, ds.t, Q, label = "Data", seriestype = :scatter, color = :red, markerstrokecolor = :red, ms = ms)
    elseif plant == "Tiller"
        Q = ds.u[2,:] # Reboiler duty is an input in Tiller
        plot!(plt_Q, ds.t, Q, label = latexstring("\\ \\textrm{Reboiler duty, SP}"), linestyle = :dash, color = :red)
    end
    
    d_adaptive_ylbl = latexstring("\\textrm{Adaptive\\ disturbance,}") * "\n" * latexstring("\\xi")
    plt_d_adaptive = plot(ylabel = d_adaptive_ylbl, ylims = ylims_d_adaptive)

    plts = (plt_yga = plt_yga, plt_Q = plt_Q, plt_d_adaptive = plt_d_adaptive)

    return plts

end

function add_to_plot(plts, ds, Y_pred, X_pred, plant, in_sample, p_fixed, lw = 2.0; label = nothing, linestyle_arg = nothing)

    # Is TlA included as state in the model?
    if haskey(p_fixed, :w_Ta)
        w_Ta = p_fixed.w_Ta
    else
        w_Ta = false
    end

    if in_sample
        linestyle = :solid
        linecolor1 = :limegreen
        linecolor2 = :blue
    else
        linestyle = :dash
        linecolor1 = :black
        linecolor2 = :red
    end

    # Override linestyle if provided as argument
    if !isnothing(linestyle_arg)
        linestyle = linestyle_arg
    end

    if isnothing(label)
        lbl_str = ", pred, " * (in_sample ? "In-sample" : "Out-of-sample")
    else
        lbl_str = label
    end

    z = extract_output(Y_pred, w_Ta, plant)

    yga = z.yga
    Na = z.Na
    Nd = z.Nd
    SRD = z.SRD
    cap_eff = z.cap_eff
    ygina = z.ygina

    plot!(plts.plt_yga, ds.t, yga, label = "yga" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
    #plot!(plts.plt_yga, ds.t, ygina, label = "ygina, data", lw = lw, color = linecolor2, linestyle = :dot)

    if plant == "DTI"
        plot!(plts.plt_TlD_HX, ds.t, z.TlD_HX, label = "TlD" * lbl_str, lw = lw, color = linecolor2, linestyle = linestyle)
        if w_Ta
            TlA_HX = z.TlA_HX
            plot!(plts.plt_TlD_HX, ds.t, TlA_HX, label = "TlA" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
        end
    end

    if plant == "DTI"
        plot!(plts.plt_Preb, ds.t, z.Preb, label = "Q" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
    end
    plot!(plts.plt_N, ds.t, Na, label = "Na" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
    plot!(plts.plt_N, ds.t, Nd, label = "Nd" * lbl_str, lw = lw, color = linecolor2, linestyle = linestyle)
    plot!(plts.plt_x, ds.t, X_pred[1,:], label = "ca" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
    plot!(plts.plt_x, ds.t, X_pred[2,:], label = "cd" * lbl_str, lw = lw, color = linecolor2, linestyle = linestyle)

    plot!(plts.plt_SRD, ds.t, SRD, label = "SRD" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
    plot!(plts.plt_cap_eff, ds.t, cap_eff, label = "Cap eff" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)

end

function add_to_plot_002(plts, ds, z, in_sample, lw = 2.0; label = nothing, linestyle_arg = nothing, line_color = nothing, t_vec = nothing)

    # Extract named outputs from z
        yga = z.yga
        Na = z.Na
        Nd = z.Nd
        SRD = z.SRD
        cap_eff = z.cap_eff
        ygina = z.ygina
        ca = z.ca
        cd = z.cd
        Tla_HX = z.Tla_HX
        Tld_HX = z.Tld_HX
        Q = z.Q
        d_adaptive = z.d_adaptive

    # Time vector
    if isnothing(t_vec)
        t_vec = ds.t
    end

    # Is TlA included as state in the model?
    # if haskey(p_fixed, :w_Ta)
    #     w_Ta = p_fixed.w_Ta
    # else
    #     w_Ta = false
    # end

    if in_sample
        linestyle = :solid
        linecolor1 = :limegreen
        linecolor2 = :blue
    else
        linestyle = :dash
        linecolor1 = :black
        linecolor2 = :red
    end

    if !isnothing(line_color)
         # Override line colors if provided as argument
        if length(line_color) == 1
            linecolor1 = line_color[1]
            linecolor2 = line_color[1]
        else
            linecolor1 = line_color[1]
            linecolor2 = line_color[2]
        end
    end

    # Override linestyle if provided as argument
    if !isnothing(linestyle_arg)
        linestyle = linestyle_arg
    end

    if isnothing(label)
        lbl_str = ", pred, " * (in_sample ? "In-sample" : "Out-of-sample")
    else
        lbl_str = label
    end

    plot!(plts.plt_yga, t_vec, yga, label = "yga" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
    #plot!(plts.plt_yga, ds.t, ygina, label = "ygina, data", lw = lw, color = linecolor2, linestyle = :dot)

    #if plant == "DTI"
        plot!(plts.plt_TlD_HX, t_vec, Tld_HX, label = "TlD" * lbl_str, lw = lw, color = linecolor2, linestyle = linestyle)
      #  if w_Ta
       #     TlA_HX = z.TlA_HX
            plot!(plts.plt_TlD_HX, t_vec, Tla_HX, label = "TlA" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
        #end
    #end

    #if plant == "DTI"
        plot!(plts.plt_Preb, t_vec, Q, label = "Q" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
    #end
    plot!(plts.plt_N, t_vec, Na, label = "Na" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
    plot!(plts.plt_N, t_vec, Nd, label = "Nd" * lbl_str, lw = lw, color = linecolor2, linestyle = linestyle)
    plot!(plts.plt_x, t_vec, ca, label = "ca" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
    plot!(plts.plt_x, t_vec, cd, label = "cd" * lbl_str, lw = lw, color = linecolor2, linestyle = linestyle)

    plot!(plts.plt_SRD, t_vec, SRD, label = "SRD" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)
    plot!(plts.plt_cap_eff, t_vec, cap_eff, label = "Cap eff" * lbl_str, lw = lw, color = linecolor1, linestyle = linestyle)

    d_adaptive_mat = stack(d_adaptive, dims = 1)
    for j in 1:size(d_adaptive_mat, 2)
         plot!(plts.plt_d_adaptive, t_vec, d_adaptive_mat[:,j], label = "d$(j)", lw = lw, linestyle = linestyle)
    end
   

end


function add_to_plot_paper(plts, ds, z, linestyle, line_color, label; adaptive = false, lw = 2.0)

    # Extract named outputs from z
    yga = z.yga
    Q = z.Q
    d_adaptive = z.d_adaptive

    yga_color, Q_color = line_color

    t_vec = ds.t

    plot!(plts.plt_yga, t_vec, yga, label =  label, lw = lw, color = yga_color, linestyle = linestyle)
    
    plot!(plts.plt_Q, t_vec, Q, label = label, lw = lw, color = Q_color, linestyle = linestyle)

    d1_color = :orange
    d2_color = :magenta
    
    if adaptive
        d_adaptive_mat = stack(d_adaptive, dims = 1)
        plot!(plts.plt_d_adaptive, t_vec, d_adaptive_mat[:,1], label = latexstring("\\ \\xi_1"), lw = lw, color = d1_color, linestyle = linestyle)
        if size(d_adaptive_mat, 2) > 1
            plot!(plts.plt_d_adaptive, t_vec, d_adaptive_mat[:,2], label = latexstring("\\ \\xi_2"), lw = lw, color = d2_color, linestyle = linestyle)
        end
    end

end
    
function grouped_bar_plot(labels, RMSE_in, RMSE_out, bar_width, config_name; ylims = (0, 0.80))
    x = 1:length(labels)+1
    bar_plt = bar(x .- bar_width/2, RMSE_in, 
                    label = "In-sample",
                    bar_width = bar_width,
                    ylabel = "RMSE",
                    xlabel = "Series",
                    title = "RMSE for config: $(config_name)",
                    ylims = ylims)
    bar!(x .+ bar_width/2, RMSE_out,
                    label = "Out-of-sample",
                    bar_width = bar_width)
    xticks!(x.+0.5, labels)                
    return bar_plt
end

function extract_output(Y, w_Ta, plant)
    
    if plant == "Tiller"
        return (yga = Y[1,:],
        Na = Y[5,:],
        Nd = Y[6,:],
        SRD = Y[7,:],
        cap_eff = Y[3,:],
        ygina = Y[2,:])
    elseif plant == "DTI"
        if w_Ta
            return (yga = Y[1,:], 
            TlA_HX = Y[2,:], 
            TlD_HX = Y[3,:], 
            Preb = Y[4,:], 
            Na = Y[5,:], 
            Nd = Y[6,:],
            cap_eff = Y[7,:],
            SRD = Y[8,:],
            ygina = Y[9,:])
        else
            return (yga = Y[1,:], 
            TlD_HX = Y[2,:], 
            Preb = Y[3,:], 
            Na = Y[4,:], 
            Nd = Y[5,:],
            cap_eff = Y[6,:],
            SRD = Y[7,:],
            ygina = Y[8,:])
        end
    else
        error("Unknown plant type: $(plant)")
    end
end


function boxplot_RMSE(RMSE_df, group_labels, campaign_group, RMSE_variable, ylims, ylbl)

    scale = 0.6
    bar_width = 0.08
    dist_groups = 1.4
    dist_sub_groups = 0.25
    dist_in_out_sample = 0.06

    number_of_groups = length(group_labels)

    sub_group_labels = repeat(["Linear\nmodel", "Nonlinear\nmodel"], outer = number_of_groups)

    group_centers = 1 .+ dist_groups .* collect(0:number_of_groups-1)
    sub_group_centers = repeat(group_centers, inner = 2) .+ repeat([-dist_sub_groups, dist_sub_groups], outer = number_of_groups) 
    xpos_box  = repeat(sub_group_centers, inner = 2) .+ repeat([-dist_in_out_sample, dist_in_out_sample], outer = number_of_groups * 2)         

    boxplt = boxplot(ylabel = ylbl, xlims = (0.0,4.0), bottom_margin = 18mm, size = scale.*(900, 600), xlim = (0.4, group_centers[end] + 0.6), ylims = ylims)

    model_type = repeat(["linear", "nonlinear"], inner = 2, outer = 3)
    sample_type = repeat(["in_sample", "out_of_sample"], outer = 6)
    #plant_group = repeat(["DTI", "DTI", "Tiller"], inner = 4)

    for i in eachindex(campaign_group)
        println("Processing campaign group: ", campaign_group[i], ", model type: ", model_type[i], ", sample type: ", sample_type[i])
        var = RMSE_df[!, Symbol(RMSE_variable)][(RMSE_df.campaign_group .== campaign_group[i]) .& (RMSE_df.model_type .== model_type[i]) .& (RMSE_df.sample_type .== sample_type[i])]
        
        if sample_type[i] == "in_sample"
            color = :limegreen
        else
            color = :blue
        end

        if i <= 2
            lbl = sample_type[i] == "in_sample" ? "In-sample" : "Out-of-sample"
        else
            lbl = ""
        end

        boxplot!(boxplt, [xpos_box[i]], var, color = color, bar_width = bar_width, label = lbl, markersize = 2.5)
    end

    xticks!(boxplt, xpos_box, fill("", length(xpos_box)))

    # Subgroup labels
    for (x, lbl) in zip(sub_group_centers, sub_group_labels)
        annotate!(boxplt, x, -0.25, text(lbl,font("Times", :black, 9)))
    end

    # Group labels
    for (x, lbl) in zip(group_centers, group_labels)
        annotate!(boxplt, x, -0.55, text(lbl,font("Times", :black, 9)))
    end

    #boxplot!(boxplt, legend = (1.4, 0.8))

    return boxplt
end


function named_output(z)

    yga = z[1,:]
    Tla_HX = z[2,:]
    Tld_HX = z[3,:]
    Q = z[4,:]
    Na = z[5,:]
    Nd = z[6,:]   
    cap_eff = z[7,:]
    SRD = z[8,:]
    ygina = z[9,:]
    ma = z[10,:]
    ca = z[11,:]
    cd = z[12,:]
    ct = z[13,:]
    d_adaptive = z[14,:]


    z_named = (yga = yga, Tla_HX = Tla_HX, Tld_HX = Tld_HX, Q = Q, 
                Na = Na, Nd = Nd, cap_eff = cap_eff, 
                SRD = SRD, ygina = ygina, ma = ma,
                ca = ca, cd = cd, ct = ct, d_adaptive = d_adaptive)

    # Create a named tuple for the outputs
    return z_named
end



function get_model_name(config_name)
    rules = [
        ["config_018A", "config_019A"] => "M1",
        ["config_018B", "config_019B"] => "M2",
        ["config_018C", "config_019C"] => "M3",
        ["config_018D", "config_019D"] => "M4",
        ["config_018E", "config_019E"] => "M5"
    ] 

    for (inputs, output) in rules
        if config_name in inputs
            return output
        end
    end

    error("Config name $(config_name) not found in rules.")
end


function boxplot_RMSE_general(RMSE_df, group, sub_group, RMSE_variable, ylims, ylbl, title)

    # Extract group and subgroup information from the provided tuples
    group_var_name, group_var_list, group_lbls = group
    sub_group_var_name, sub_group_var_list, sub_group_lbls = sub_group

    scale = 0.6
    bar_width = 0.08
    dist_groups = 1.4
    dist_sub_groups = 0.25
    dist_in_out_sample = 0.06

    n_groups = length(group_lbls)
    n_subgroups = length(sub_group_lbls) # subgroups pr. group

    group_centers = 1 .+ dist_groups .* collect(0:n_groups-1)


    sub_group_centers_rel = collect(-(n_subgroups-1)/2 * dist_sub_groups:dist_sub_groups:(n_subgroups-1)/2 * dist_sub_groups)
    sub_group_centers = repeat(group_centers, inner = n_subgroups) .+ repeat(sub_group_centers_rel, outer = n_groups)

    xpos_box  = repeat(sub_group_centers, inner = 2) .+ repeat([-dist_in_out_sample, dist_in_out_sample], outer = n_groups * n_subgroups)         

    boxplt = boxplot(ylabel = ylbl, xlims = (0.0,4.0), bottom_margin = 18mm, size = scale.*(900, 600), xlim = (0.4, group_centers[end] + 0.6), ylims = ylims, title = title)

    N_total = n_groups * n_subgroups * 2
    i = 0
    for group_var in group_var_list
        for sub_group_var in sub_group_var_list
            for sample in ["in-sample", "out-of-sample"]
                i += 1
                #println("Processing group: ", group_var, ", subgroup: ", sub_group_var, ", sample type: ", sample)
                var = RMSE_df[!, RMSE_variable][RMSE_df[!, group_var_name] .== group_var .&& RMSE_df[!, sub_group_var_name] .== sub_group_var .&& RMSE_df.sample_type .== sample]
                
                if isempty(var)
                    println("Warning: No data found for group: ", group_var, ", subgroup: ", sub_group_var, ", sample type: ", sample)
                    continue
                end
                if any(isnan.(var))
                    println("Warning: NaN values found for group: ", group_var, ", subgroup: ", sub_group_var, ", sample type: ", sample)
                    continue
                end

                if sample == "in-sample"
                    color = :limegreen
                else
                    color = :blue
                end

                if i <= 2
                    #lbl = sample == "in-sample" ? "In-sample" : "Out-of-sample"
                    lbl = ""
                else
                    lbl = ""
                end

                boxplot!(boxplt, [xpos_box[i]], var, color = color, bar_width = bar_width, label = lbl, markersize = 2.5)
            end
        end
    end

    xticks!(boxplt, xpos_box, fill("", length(xpos_box)))

    # Subgroup labels
    
    for (x, lbl) in zip(sub_group_centers, repeat(sub_group_lbls, outer = n_groups))
        annotate!(boxplt, x, -0.25, text(lbl,font("Times", :black, 9)))
    end

    # Group labels
    for (x, lbl) in zip(group_centers, group_lbls)
        annotate!(boxplt, x, -0.55, text(lbl,font("Times", :black, 9)))
    end

    #boxplot!(boxplt, legend = (1.4, 0.8))

    return boxplt
end



function boxplot_RMSE_general_adap(RMSE_df, group, sub_group, RMSE_variable, ylims, ylbl, title)

    # Extract group and subgroup information from the provided tuples
    group_var_name, group_var_list, group_lbls = group
    sub_group_var_name, sub_group_var_list, sub_group_lbls = sub_group

    scale = 0.6
    bar_width = 0.08
    dist_groups = 1.4
    dist_sub_groups = 0.25
    dist_in_out_sample = 0.06

    n_groups = length(group_lbls)
    n_subgroups = length(sub_group_lbls) # subgroups pr. group

    group_centers = 1 .+ dist_groups .* collect(0:n_groups-1)


    sub_group_centers_rel = collect(-(n_subgroups-1)/2 * dist_sub_groups:dist_sub_groups:(n_subgroups-1)/2 * dist_sub_groups)
    sub_group_centers = repeat(group_centers, inner = n_subgroups) .+ repeat(sub_group_centers_rel, outer = n_groups)

    xpos_box  = repeat(sub_group_centers, inner = 2) .+ repeat([-dist_in_out_sample, dist_in_out_sample], outer = n_groups * n_subgroups)         

    boxplt = boxplot(ylabel = ylbl, xlims = (0.0,4.0), bottom_margin = 18mm, size = scale.*(900, 600), xlim = (0.4, group_centers[end] + 0.6), ylims = ylims, title = title)

    N_total = n_groups * n_subgroups * 2
    i = 0
    for group_var in group_var_list
        for sub_group_var in sub_group_var_list
            for sample in ["pred", "adaptive"]
                i += 1
                #println("Processing group: ", group_var, ", subgroup: ", sub_group_var, ", sample type: ", sample)
                if sample == "pred"
                    RMSE_variable_i = RMSE_variable
                else
                    RMSE_variable_i = Symbol(string(RMSE_variable)*"_adap")
                end
                
                # Extract all vars that match the current group, subgroup, and sample type
                var = RMSE_df[!, RMSE_variable_i][RMSE_df[!, group_var_name] .== group_var .&& RMSE_df[!, sub_group_var_name] .== sub_group_var .&& RMSE_df.sample_type .== "out-of-sample"]
                
                if isempty(var)
                    println("Warning: No data found for group: ", group_var, ", subgroup: ", sub_group_var, ", sample type: ", sample)
                    continue
                end
                if any(isnan.(var))
                    println("Warning: NaN values found for group: ", group_var, ", subgroup: ", sub_group_var, ", sample type: ", sample)
                    continue
                end

                if sample == "pred"
                    color = :blue
                else
                    color = :limegreen
                end

                if i <= 2
                    #lbl = sample == "in-sample" ? "In-sample" : "Out-of-sample"
                    lbl = ""
                else
                    lbl = ""
                end

                boxplot!(boxplt, [xpos_box[i]], var, color = color, bar_width = bar_width, label = lbl, markersize = 2.5)
            end
        end
    end

    xticks!(boxplt, xpos_box, fill("", length(xpos_box)))

    # Subgroup labels
    
    for (x, lbl) in zip(sub_group_centers, repeat(sub_group_lbls, outer = n_groups))
        annotate!(boxplt, x, -0.25, text(lbl,font("Times", :black, 9)))
    end

    # Group labels
    for (x, lbl) in zip(group_centers, group_lbls)
        annotate!(boxplt, x, -0.55, text(lbl,font("Times", :black, 9)))
    end

    #boxplot!(boxplt, legend = (1.4, 0.8))

    return boxplt
end

function barplot_RMSE_general_adap(RMSE_df, group, sub_group, RMSE_variables, ylims, ylbl, title)

    group_var_name, group_var_list, group_lbls = group
    sub_group_var_name, sub_group_var_list, sub_group_lbls = sub_group

    scale = 0.6
    bar_width = 0.08
    dist_groups = 1.4
    dist_sub_groups = 0.25
    dist_in_out_sample = 0.06

    n_groups = length(group_lbls)
    n_subgroups = length(sub_group_lbls)

    group_centers = 1 .+ dist_groups .* collect(0:n_groups-1)
    sub_group_centers_rel = collect(-(n_subgroups-1)/2 * dist_sub_groups:dist_sub_groups:(n_subgroups-1)/2 * dist_sub_groups)
    sub_group_centers = repeat(group_centers, inner = n_subgroups) .+ repeat(sub_group_centers_rel, outer = n_groups)
    xpos_box = repeat(sub_group_centers, inner = 2) .+ repeat([-dist_in_out_sample, dist_in_out_sample], outer = n_groups * n_subgroups)

    barplt = bar(ylabel = ylbl, bottom_margin = 18mm, size = scale .* (900, 600),
                 xlim = (0.4, group_centers[end] + 0.6), ylims = ylims, title = title)

    i = 0
    for group_var in group_var_list
        for sub_group_var in sub_group_var_list
            for sample in ["pred", "adaptive"]
                i += 1

                # Collect squared errors across all variables, then take a single sqrt(mean(...))
                all_sq_errors = Float64[]
                skip = false

                for RMSE_variable in RMSE_variables
                    RMSE_variable_i = sample == "pred" ? RMSE_variable : Symbol(string(RMSE_variable) * "_adap")

                    var = RMSE_df[!, RMSE_variable_i][
                        RMSE_df[!, group_var_name]     .== group_var .&&
                        RMSE_df[!, sub_group_var_name] .== sub_group_var .&&
                        RMSE_df.sample_type            .== "out-of-sample"
                    ]

                    if isempty(var) || any(isnan.(var))
                        println("Warning: skipping group=$group_var, subgroup=$sub_group_var, sample=$sample, variable=$RMSE_variable")
                        skip = true
                        break
                    end

                    append!(all_sq_errors, var .^ 2)
                end

                skip && continue

                RMSE_tot = sqrt(mean(all_sq_errors))
                color    = sample == "pred" ? :blue : :limegreen
                lbl      = i <= 2 ? (sample == "pred" ? "Predicted" : "Adaptive") : ""

                bar!(barplt, [xpos_box[i]], [RMSE_tot], color = color,
                     bar_width = bar_width, label = lbl, linewidth = 0)
            end
        end
    end

    xticks!(barplt, xpos_box, fill("", length(xpos_box)))

    for (x, lbl) in zip(sub_group_centers, repeat(sub_group_lbls, outer = n_groups))
        annotate!(barplt, x, -0.25, text(lbl, font("Times", :black, 9)))
    end

    for (x, lbl) in zip(group_centers, group_lbls)
        annotate!(barplt, x, -0.55, text(lbl, font("Times", :black, 9)))
    end

    return barplt
end

function mix_with_white(hex::String, x::Real)
    c = parse(RGB, hex)
    t = x / 100
    return RGB(
        t + (1 - t) * red(c),
        t + (1 - t) * green(c),
        t + (1 - t) * blue(c)
    )
end

function remove_y_ax_lbls_and_ticks!(plt)
    ytick_i = yticks(plt)
    y_tick_i_labels = ["" for _ in 1:length(ytick_i[1][1])]
    y_tick_i_new = (ytick_i[1][1], y_tick_i_labels)
    plt = plot!(plt, yticks = y_tick_i_new, ylabel = "")
    return plt
end

function remove_x_ax_lbls_and_ticks!(plt)
    xtick_i = xticks(plt)
    x_tick_i_labels = ["" for _ in 1:length(xtick_i[1][1])]
    x_tick_i_new = (xtick_i[1][1], x_tick_i_labels)
    plt = plot!(plt, xticks = x_tick_i_new, xlabel = "")
    return plt
end 