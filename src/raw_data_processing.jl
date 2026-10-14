# ------------------------------------------------------------------------------------------
# Processing of the raw pilot-plant logs (campaign 4, Nov-Dec 2025) into the data series used
# for model training and validation.
#
# The functions follow the original data analysis of the DTI pilot plant
# (DTI_pilot_data_analysis: src/load_and_process.jl, src/set_point_write.jl, src/util.jl and
# driver/driver_ss_analysis.jl). The original analysis processed all four measurement campaigns
# together; here only campaign 4 is processed. The resulting series are identical to the
# original ones, except that the column `ygA_in_SP` (a mean over all four campaigns, not used
# by the model) is left out.
# ------------------------------------------------------------------------------------------

# Raw log files of campaign 4 (exported from the plant logger, tab separated, decimal comma)
const RAW_DATA_FILES = [
    "111 - 2025-11-24 Default.txt",
    "112 - 2025-11-27 Default.txt",
    "113 - 2025-11-27 Default.txt",
    "116 - 2025-12-01 Default.txt",
    "117 - 2025-12-03 Default.txt",
]

# Time origin of the time column `t` [h]: the first sample of the DTI measurement campaigns
# (27 May 2024). Kept from the original analysis so that the series are reproduced exactly.
const T_ORIGIN = DateTime(2024, 5, 27, 13, 30, 50)

# From src/util.jl (DTI_pilot_data_analysis)
function takeevery(iter, n)
    return (x for (i, x) in enumerate(iter) if i % n == 0)
end

# From src/util.jl (DTI_pilot_data_analysis): convert a column from String to Float64
function convert_column!(df::DataFrame, col::Symbol)
    T = eltype(df[!, col])

    if T <: AbstractString || T <: Union{Missing, AbstractString}
        df[!, col] = parse.(Float64, replace.(df[!, col], "," => "."))
    end
end

# Centered moving average of x[i] in the range i±(n-1)/2 (the averaging of EasyFit.movingaverage,
# which was used in the original analysis)
function easyfit_movavg(X::AbstractArray{T}, n::Int) where {T<:Real}
    if !isodd(n)
        n = n + 1
    end
    delta = div(n - 1, 2)
    nX = length(X)
    x = similar(X)
    for i in 1:nX
        jmin = max(i - delta, 1)
        jmax = min(i + delta, nX)
        x[i] = zero(T)
        for j in jmin:jmax
            x[i] = x[i] + X[j]
        end
        x[i] = x[i] / (jmax - jmin + 1)
    end
    return x
end

"""
    load_raw_data_files(raw_data_folder)

Load and concatenate the raw log files of campaign 4 (load_raw_data_files in
DTI_pilot_data_analysis/src/load_and_process.jl).
"""
function load_raw_data_files(raw_data_folder)
    df_raw = DataFrame()

    for file in RAW_DATA_FILES
        df = DataFrame(takeevery(CSV.File(joinpath(raw_data_folder, file); delim = '\t'), 1))
        df_raw = vcat(df_raw, df)
    end

    # "," => "." in time column and remove milliseconds for parsing
    df_raw[:, "PC time"] = replace.(df_raw[:, "PC time"], "," => ".")
    clean = first.(split.(df_raw[!, "PC time"], '.'))
    df_raw[!, "PC time"] = DateTime.(clean, dateformat"dd-mm-yyyy HH:MM:SS")

    # Convert all measurement columns from strings with decimal comma to Float64
    for col_name in names(df_raw[:, 3:end-1])
        convert_column!(df_raw, Symbol(col_name))
    end

    df_raw.time = df_raw[!, "PC time"]

    # Fix the encoding of Danish characters in the column names
    rename!(df_raw) do col_name
        col_name = replace(col_name, "[�C]" => "[°C]")
        col_name = replace(col_name, "v�d" => "våd")
        col_name = replace(col_name, "t�r" => "tør")
        col_name = replace(col_name, "Tr�k" => "Tryk")
        col_name = replace(col_name, "R�g" => "Røg")
        col_name = replace(col_name, "Flow_f�r" => "Flow_før")
    end

    return df_raw
end

"""
    compute_data_analysis_variables(df)

Compute the process variables from the raw signals (compute_data_analysis_variables in
DTI_pilot_data_analysis/src/load_and_process.jl).
"""
function compute_data_analysis_variables(df::DataFrame)

    # Constants
    R = 8.3145 # J/(mol·K)
    M_CO2 = 44.01 # g/mol
    g2kg = 1e-3 # g to kg
    C2K = 273.15 # Celsius to Kelvin
    s2h = 1.0 / 3600.0
    kJ2MJ = 1.0 / 1000.0

    # Time
    time = df[:, "time"]
    t = Dates.value.(time .- T_ORIGIN) ./ 1000.0 ./ 3600.0  # Time in hours
    dt = diff(t)

    # Absorber gas properties
    # Inlet
    ygA_in = df[:, "CO2_abs_in - [%]"]
    TginA = df[:, "A020_BT01 - [°C]"]
    cginA_tot = 101325 ./ (R .* (TginA .+ C2K))        # By ideal gas law [mol/m3]
    cginA = ygA_in ./ 100 .* cginA_tot                 # CO2 concentration in inlet gas [mol/m3]
    m_ginA = df[:, "Flow_før abs - [kg/h]"]
    rho_ginA = 1.3014 # Inlet gas density [kg/m3]
    FginA = m_ginA ./ rho_ginA
    mginA_CO2 = cginA .* M_CO2 .* g2kg .* FginA        # CO2 mass flow in inlet gas [kg/h]

    # Outlet
    ygA = df[:, "CO2_tør_abs - [%]"]
    TgA = df[:, "A020_BT02 - [°C]"]
    cgA_tot = 101325 ./ (R .* (TgA .+ C2K))            # [mol/m3]
    cgA = ygA ./ 100 .* cgA_tot
    mga_CO2 = cgA .* M_CO2 .* g2kg .* FginA            # CO2 mass flow in outlet gas [kg/h]

    # Solvent crossover flow properties
    Fco = df[:, "A050_BF01 - [m3/h]"]
    rho_liq = 1020 # Liquid density [kg/m3]

    # Reboiler properties
    P_reb_acc = df[:, "Damp_kedel - [kWh]"]
    P_reb_raw = diff(P_reb_acc) ./ dt
    P_reb = easyfit_movavg(P_reb_raw, 10)
    P_reb_valve = df[:, "A060_QN01_PV - [% open]"]

    # Temperatures
    TlD_HX_out = df[:, "A060_BT02 - [°C]"] # Reboiler HX outlet [C]
    TlD = df[:, "A050_BT02 - [°C]"]        # Desorber temperature [C]
    TlA = df[:, "A010_BT01 - [°C]"]        # Absorber temperature [C]
    TlA_HX_out = df[:, "A010_BT02 - [°C]"] # Absorber HX outlet [C]

    # Key performance indicators
    m_captured_CO2 = mginA_CO2 .- mga_CO2                       # Captured CO2 mass flow [kg/h]
    SRD = (P_reb ./ s2h .* kJ2MJ) ./ m_captured_CO2[2:end]      # Specific reboiler duty [MJ/kg]
    LG_ratio = Fco .* rho_liq ./ m_ginA
    cap_eff = m_captured_CO2 ./ mginA_CO2 * 100

    df_analysis = DataFrame(
        time = time,
        t = t,
        ygA_in = ygA_in,
        TginA = TginA,
        cginA_tot = cginA_tot,
        cginA = cginA,
        m_ginA = m_ginA,
        FginA = FginA,
        mginA_CO2 = mginA_CO2,
        ygA = ygA,
        TgA = TgA,
        cgA_tot = cgA_tot,
        cgA = cgA,
        mga_CO2 = mga_CO2,
        Fco = Fco,
        P_reb = [NaN; P_reb],  # Align lengths
        P_reb_valve = P_reb_valve,
        TlD_HX_out = TlD_HX_out,
        TlD = TlD,
        TlA = TlA,
        TlA_HX_out = TlA_HX_out,
        m_captured_CO2 = m_captured_CO2,
        SRD = [NaN; SRD],      # Align lengths
        LG_ratio = LG_ratio,
        cap_eff = cap_eff
    )

    return df_analysis
end

# ------------------------------------------------------------------------------------------
# Test plan of campaign 4: 36 load points (series 1-36 of the test plan, run in the order
# `series_order`), with the start times of the load points and their set points
# (write_set_points in DTI_pilot_data_analysis/src/set_point_write.jl)
# ------------------------------------------------------------------------------------------
const SP_START_TIME_CAMP_04 = [
    DateTime("2025-11-24T09:00")                 # Series 1
    DateTime("2025-11-24T12:43")                 # Series 2
    DateTime("2025-11-24T14:46:30") - Minute(8)  # Series 3
    DateTime("2025-11-24T16:56:30")              # Series 4
    DateTime("2025-11-24T18:52") - Minute(8)     # Series 5
    DateTime("2025-11-24T20:43") - Minute(8)     # Series 6
    DateTime("2025-11-24T22:55")                 # Series 7
    DateTime("2025-11-25T09:01") - Minute(8)     # Series 8
    DateTime("2025-11-25T10:40") - Minute(8)     # Series 9
    DateTime("2025-11-25T14:27")                 # Series 13
    DateTime("2025-11-25T16:25") - Minute(8)     # Series 14
    DateTime("2025-11-25T18:25:30") - Minute(8)  # Series 15
    DateTime("2025-11-25T20:18")                 # Series 16
    DateTime("2025-11-26T07:51") - Minute(8)     # Series 17
    DateTime("2025-11-26T09:43") - Minute(8)     # Series 18
    DateTime("2025-11-26T14:00")                 # Series 19
    DateTime("2025-11-26T16:24:30") - Minute(8)  # Series 20
    DateTime("2025-11-26T19:40") - Minute(8)     # Series 21
    DateTime("2025-11-26T22:47")                 # Series 22
    DateTime("2025-11-27T17:25:30")              # Series 23
    DateTime("2025-11-27T19:57:30")              # Series 24
    DateTime("2025-12-01T11:30")                 # Series 10
    DateTime("2025-12-01T14:00:30") - Minute(8)  # Series 11
    DateTime("2025-12-01T16:33") - Minute(8)     # Series 12
    DateTime("2025-12-01T18:44:30")              # Series 25
    DateTime("2025-12-02T08:58") - Minute(8)     # Series 26
    DateTime("2025-12-02T10:41") - Minute(8)     # Series 27
    DateTime("2025-12-02T12:49")                 # Series 28
    DateTime("2025-12-02T15:16") - Minute(8)     # Series 29
    DateTime("2025-12-02T17:26") - Minute(8)     # Series 30
    DateTime("2025-12-02T19:31")                 # Series 31
    DateTime("2025-12-03T07:46") - Minute(8)     # Series 32
    DateTime("2025-12-03T09:31") - Minute(8)     # Series 33
    DateTime("2025-12-03T12:00")                 # Series 34
    DateTime("2025-12-03T13:53:30") - Minute(8)  # Series 35
    DateTime("2025-12-03T16:02") - Minute(8)     # Series 36
]

const SERIES_ORDER_CAMP_04 = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                              13, 14, 15, 16, 17, 18,
                              19, 20, 21, 22, 23, 24,
                              10, 11, 12,
                              25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

# Set points of the 36 load points of the test plan
const SP_TD_VAL_CAMP_04 = [95, 90, 85, 85, 90, 95, 85, 90, 95, 95, 90, 85,
                           95, 90, 85, 85, 90, 95, 95, 90, 85, 85, 90, 95,
                           85, 90, 95, 95, 90, 85, 85, 90, 95, 95, 90, 85]
const SP_MGINA_VAL_CAMP_04 = repeat([150, 100, 50], inner = 12)
const SP_FCO_VAL_CAMP_04 = repeat([0.8, 0.6, 0.4, 0.2], inner = 3, outer = 3)

# Absorber temperature set point. It was not changed in campaign 4; the value of the previous
# campaign (35 °C, set 7 March 2025) is kept as in the original analysis. Not used by the model.
const SP_TA_CAMP_04 = 35.0

# Start times of the load points in the planned test sequence, used for the target columns
# (load_ss_periods in DTI_pilot_data_analysis/src/load_and_process.jl)
const LOADPOINT_START_TIMES_CAMP_04 = [
    DateTime("2025-11-24T09:00"), DateTime("2025-11-24T12:51"), DateTime("2025-11-24T14:46:30"),
    DateTime("2025-11-24T16:56:30"), DateTime("2025-11-24T18:52"), DateTime("2025-11-24T20:43"),
    DateTime("2025-11-24T22:55"), DateTime("2025-11-25T09:01"), DateTime("2025-11-25T10:40"),
    DateTime("2025-11-25T14:27"), DateTime("2025-11-25T16:25"), DateTime("2025-11-25T18:25:30"),
    DateTime("2025-11-25T20:18"), DateTime("2025-11-26T07:51"), DateTime("2025-11-26T09:43"),
    DateTime("2025-11-26T14:00"), DateTime("2025-11-26T16:24:30"), DateTime("2025-11-26T19:40"),
    DateTime("2025-11-26T22:47"), DateTime("2025-11-27T17:25:30"), DateTime("2025-11-27T19:57:30"),
    DateTime("2025-12-01T11:30"), DateTime("2025-12-01T14:00:30"), DateTime("2025-12-01T16:33"),
    DateTime("2025-12-01T18:44:30"), DateTime("2025-12-02T08:58"), DateTime("2025-12-02T10:41"),
    DateTime("2025-12-02T12:49"), DateTime("2025-12-02T15:16"), DateTime("2025-12-02T17:26"),
    DateTime("2025-12-02T19:31"), DateTime("2025-12-03T07:46"), DateTime("2025-12-03T09:31"),
    DateTime("2025-12-03T12:00"), DateTime("2025-12-03T13:53:30"), DateTime("2025-12-03T16:02"),
]

"""
    add_set_points!(df)

Add the set point columns (m_ginA_SP, Fco_SP, TlD_HX_out_SP, TlA_HX_out_SP) and the target
columns of the test plan (m_ginA_target, Fco_target, TlD_HX_out_target) to df. Returns the
times of the load point changes.
"""
function add_set_points!(df::DataFrame)
    df.m_ginA_SP = fill(NaN, nrow(df))
    df.Fco_SP = fill(NaN, nrow(df))
    df.TlD_HX_out_SP = fill(NaN, nrow(df))
    df.TlA_HX_out_SP = fill(SP_TA_CAMP_04, nrow(df))

    for (i, t0) in enumerate(SP_START_TIME_CAMP_04)
        series_idx = SERIES_ORDER_CAMP_04[i]
        idx = findall(x -> x >= t0, df.time)
        df.TlD_HX_out_SP[idx] .= SP_TD_VAL_CAMP_04[series_idx]
        df.m_ginA_SP[idx] .= SP_MGINA_VAL_CAMP_04[series_idx]
        df.Fco_SP[idx] .= SP_FCO_VAL_CAMP_04[series_idx]
    end

    df.m_ginA_target = fill(NaN, nrow(df))
    df.Fco_target = fill(NaN, nrow(df))
    df.TlD_HX_out_target = fill(NaN, nrow(df))
    for (i, start_time) in enumerate(LOADPOINT_START_TIMES_CAMP_04)
        end_time = i < length(LOADPOINT_START_TIMES_CAMP_04) ? LOADPOINT_START_TIMES_CAMP_04[i+1] : df.time[end]
        idx = findall(t -> t >= start_time && t < end_time, df.time)
        series_idx = SERIES_ORDER_CAMP_04[i]
        df[idx, :m_ginA_target] .= SP_MGINA_VAL_CAMP_04[series_idx]
        df[idx, :Fco_target] .= SP_FCO_VAL_CAMP_04[series_idx]
        df[idx, :TlD_HX_out_target] .= SP_TD_VAL_CAMP_04[series_idx]
    end

    # Load point changes: change k happens at the start of load point k+1
    return SP_START_TIME_CAMP_04[2:end]
end

# ------------------------------------------------------------------------------------------
# Dynamic data series: each series covers one or more consecutive load point changes (step
# tests with valid dynamic responses), from 25 min before the first change to `duration_after`
# after the last change (driver_ss_analysis.jl in DTI_pilot_data_analysis, groups 14-23).
# ------------------------------------------------------------------------------------------
const DYNAMIC_SERIES_CAMP_04 = [
    (load_point_changes = [3, 4, 5],       duration_after = Minute(100)),   # Series 1
    (load_point_changes = [7, 8],          duration_after = Minute(40)),    # Series 2
    (load_point_changes = [10, 11, 12],    duration_after = Minute(100)),   # Series 3
    (load_point_changes = [13, 14],        duration_after = Minute(100)),   # Series 4
    (load_point_changes = [16, 17, 18],    duration_after = Minute(100)),   # Series 5
    (load_point_changes = [21, 22, 23, 24], duration_after = Minute(100)),  # Series 6
    (load_point_changes = [25, 26],        duration_after = Minute(100)),   # Series 7
    (load_point_changes = [28, 29, 30],    duration_after = Minute(100)),   # Series 8
    (load_point_changes = [31, 32],        duration_after = Minute(120)),   # Series 9
    (load_point_changes = [34, 35],        duration_after = Minute(100)),   # Series 10
]

"""
    cut_dynamic_series(df, load_point_change_times)

Cut the processed data into the 10 dynamic data series.
"""
function cut_dynamic_series(df::DataFrame, load_point_change_times)
    series = DataFrame[]
    for s in DYNAMIC_SERIES_CAMP_04
        t0 = load_point_change_times[s.load_point_changes[1]] - Minute(25)
        tf = load_point_change_times[s.load_point_changes[end]] + s.duration_after
        push!(series, df[findall(x -> x >= t0 && x <= tf, df.time), :])
    end
    return series
end
