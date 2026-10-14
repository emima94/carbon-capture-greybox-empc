# Functions for loading and processing the pilot-plant data from the EMPC demonstration, and for
# the state estimation in each control interval of the closed-loop EMPC.
# Verbatim copies of the functions of the same name in src/util_closed_loop_MPC.jl of the original project
# (copied so that the figure drivers do not depend on the DuckDB and plotting code of that file).

function load_csv_data_and_process(csv_path::String)

    df_long = CSV.read(csv_path, DataFrame, dateformat="yyyy-mm-dd HH:MM:SS.zzz")

    df_long[!,:ts] = String.(df_long[!,:ts])

    df = choose_tags_and_reshape_long_df(df_long)

    df = process_data(df)

    return df

end

function choose_tags_and_reshape_long_df(df_long::DataFrame)

    df_wide = unstack(df_long, :ts, :sensor, :value)
    # df_wide.
    for name in names(df_wide)
        #println(name)
    end
    tag_load = ["CO2_abs_in - [%]", "Flow_før abs - [kg/h]", "CO2_tør_abs - [%]", "A010_BT02 - [°C]", 
                    "A020_BT01 - [°C]", "A020_BT02 - [°C]", "A060_BT02 - [°C]", "Damp_kedel - [kWh]", 
                    "A050_BF01 - [m3/h]", "Flue_inlet_temp - [°C]", "Røg - [°C]", "Effekt - [kW]"]
    var_names = [:ygina, :mgina, :yga, :Tla_HX, :Tgina, :Tga, :Tld_HX, :Q_acc, :F, :Tg_boiler, :Tg_boiler_2, :boiler_power]

    df = DataFrame()

    for (i, tag) in enumerate(tag_load)

        # col = df_wide[!, tag]
        # n_missing = count(ismissing, col)
        # if n_missing > 0
        #     println("  → $n_missing missing value(s) found in tag: ", tag)
        # end

        if tag in names(df_wide)
            #println("Loading tag: ", tag)

            df[!, var_names[i]] = Float64.(coalesce.(df_wide[!, tag], NaN))
            #df[!, var_names[i]] = Float64.(df_wide[!, tag])
        else
            println("Warning: Tag $(tag) not found in data.")
            df[!, var_names[i]] = fill(NaN, nrow(df_wide))  # optional fallback
        end
    end
    # add time stamps
    df.time = parse_zdt.(df_wide.ts)

    # Remove rows with NaN values
    df_clean = filter(row -> all(x -> !(x isa Float64 && isnan(x)), values(row)), df)
    #println("Dropped $(nrow(df) - nrow(df_clean)) rows with NaN values")

    return df_clean

end

function parse_zdt(s::String)
    #fmt = contains(s, ".") ? dateformat"yyyy-mm-dd HH:MM:SS.sssssszzz" : dateformat"yyyy-mm-dd HH:MM:SSzzz"
    fmt = contains(s, ".") ? dateformat"yyyy-mm-dd HH:MM:SS.sssssszzz" : dateformat"yyyy-mm-dd HH:MM:SSzzz"
    #return ZonedDateTime(s, fmt)
    return DateTime(s, fmt)
end

function process_data(df)

    # Constants
    P = 101325 # Pressure in Pa
    R = 8.314 # Gas constant in J/(mol*K)
    C2K = 273.15 # Celsius to Kelvin conversion
    rho_gina = 1.3014 # Inlet gas density [kg/m3]
    g2kg = 1e-3
    M_CO2 = 44.01 # Molar mass of CO2 in g/mol
    s2h = 1.0 / 3600.0 # Seconds to hours conversion
    kJ2MJ = 1e-3 # kJ to MJ conversion

    # Post process data
    # Inlet flue gas conditions
    cgina_tot = P ./ (R .* (df.Tgina .+ C2K))        # By ideal gas law [mol/m3]
    cga_tot = P ./ (R .* (df.Tga .+ C2K))        # By ideal gas law [mol/m3]
    df.cgina = df.ygina ./ 100 .* cgina_tot
    df.Fgina = df.mgina ./ rho_gina
    df.cga = df.yga ./ 100 .* cga_tot
    # Capture rate
    df.Na = df.Fgina .* (df.cgina - df.cga) # Capture rate [mol/h], mass balance CO2 gas phase, absorber (assuming steady state)

    # Reboiler duty
    t0 = df.time[1]
    t = Dates.value.(df.time .- t0) ./ 1000.0 ./ 3600 # Convert time to hours
    dt = diff(t)
    Q_raw = diff(df.Q_acc) ./ dt # Convert from kWh per time step to kW
    window_size = 50
    Q_smooth = moving_average(Q_raw, window_size) # Smooth with moving average
    df.Q = [Q_smooth; Q_smooth[end]] 
    df.t = t

    # CO2 mass flow rates in/out of absorber
    df.mgina_CO2 = df.cgina .* df.Fgina * M_CO2 * g2kg 
    #df.mga_CO2 = df.cga .* df.Fgina * M_CO2 * g2kg
    df.mga_CO2 = df.Na .* M_CO2 * g2kg # Capture rate in mass flow rate units [kg/h]

    # Capture rate CO2 [kg/h]
    #df.m_cap_CO2 = df.Na * M_CO2 * g2kg

    # Specific reboiler duty [MJ/kg CO2 captured]
    df.SRD = (df.Q ./ s2h .* kJ2MJ) ./ (df.mga_CO2)

    # Capture efficiency
    df.cap_eff = df.mga_CO2 ./ df.mgina_CO2 * 100

    return df
end

function get_control(t::Float64, time_control::Vector{Float64},
                     u::Union{Vector{Vector{Float64}}, Vector{Float64}},
                     u_prev::Union{Vector{Float64}, Float64})
    idx = findlast(tc -> tc <= t, time_control)
    return idx === nothing ? u_prev : u[idx]
end


function get_control_interval_rows(df, time_previous, time_current)
    if df.time[end] < time_previous  
        idx_0_current_interval = length(df.time)
    else
        idx_0_current_interval = findfirst(df.time .>= time_previous) - 1
    end

    if df.time[end] < time_current
        idx_f_current_interval = length(df.time)
    else
        idx_f_current_interval = findfirst(df.time .>= time_current) - 1
    end


    df_current_interval = df[idx_0_current_interval:idx_f_current_interval, :]

    t_current_interval = ([time_previous; df.time[idx_0_current_interval+1:idx_f_current_interval]; time_current] .- time_previous) ./ Hour(1) # Time in hours since start of current control interval

    return df_current_interval, t_current_interval
    
end

function update_CO2_accumulated!(Mga, Mgina, df_current_interval, t_current_interval)

    dt_current_interval = diff(t_current_interval)

    mga = df_current_interval.mga_CO2
    mgina_CO2 = df_current_interval.mgina_CO2

    Mga += sum(mga .* dt_current_interval)
    Mgina += sum(mgina_CO2 .* dt_current_interval)

    return Mga, Mgina
    
end

function get_dist_and_meas_val_current_interval(df_current_interval)

    dist_var = [:cgina, :Fgina]
    meas_var = [:yga, :Tld_HX, :Q]

    # Get disturbance values for current control interval
    dist = Matrix(df_current_interval[!, dist_var])'

    # Get measurement values for current control interval
    y_meas = Matrix(df_current_interval[!, meas_var])'

    return dist, y_meas
end

# Version of choose_tags_and_reshape_long_df used by the controller during the closed-loop tests
# (commit a1ceaea, 15 April 2026), which loaded 9 sensor tags. The boiler tags were added later for
# the evaluation of the tests. Rows with a missing value in a loaded tag are dropped, so the two
# versions give slightly different data.
function choose_tags_and_reshape_long_df_controller(df_long::DataFrame)

    df_wide = unstack(df_long, :ts, :sensor, :value)

    # for name in names(df_wide)
    #     println(name)
    # end
    tag_load = ["CO2_abs_in - [%]", "Flow_før abs - [kg/h]", "CO2_tør_abs - [%]", "A010_BT02 - [°C]", "A020_BT01 - [°C]", "A020_BT02 - [°C]", "A060_BT02 - [°C]", "Damp_kedel - [kWh]", "A050_BF01 - [m3/h]"]
    var_names = [:ygina, :mgina, :yga, :Tla_HX, :Tgina, :Tga, :Tld_HX, :Q_acc, :F]

    df = DataFrame()

    for (i, tag) in enumerate(tag_load)

        # col = df_wide[!, tag]
        # n_missing = count(ismissing, col)
        # if n_missing > 0
        #     println("  → $n_missing missing value(s) found in tag: ", tag)
        # end

        if tag in names(df_wide)
            #println("Loading tag: ", tag)

            df[!, var_names[i]] = Float64.(coalesce.(df_wide[!, tag], NaN))
            #df[!, var_names[i]] = Float64.(df_wide[!, tag])
        else
            println("Warning: Tag $(tag) not found in data.")
            df[!, var_names[i]] = fill(NaN, nrow(df_wide))  # optional fallback
        end
    end
    # add time stamps
    df.time = parse_zdt.(df_wide.ts)

    # Remove rows with NaN values
    df_clean = filter(row -> all(x -> !(x isa Float64 && isnan(x)), values(row)), df)
    #println("Dropped $(nrow(df) - nrow(df_clean)) rows with NaN values")

    return df_clean

end
