# ------------------------------------------------------------------------------------------
# Process the raw pilot-plant logs into the 10 data series used for model training and
# validation (Section 2.2, Table 2)
#
# Steps:
#   1) Load the raw logs of measurement campaign 4 (Nov-Dec 2025) from data/raw_pilot_plant/
#   2) Compute the process variables (gas concentrations and flows, CO2 mass flows, reboiler
#      duty from the accumulated boiler energy, capture efficiency, ...)
#   3) Add the set points of the test plan
#   4) Cut the data into 10 dynamic series around the step changes of the set points
#
# Output: data/pilot_plant_series/series_1.csv ... series_10.csv
# ------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
include(joinpath(@__DIR__, "..", "src", "raw_data_processing.jl"))

println("Loading raw data...")
df_raw = load_raw_data_files(joinpath(DATA_DIR, "raw_pilot_plant"))
println("  $(nrow(df_raw)) samples from $(df_raw.time[1]) to $(df_raw.time[end])")

println("Computing process variables and adding set points...")
df = compute_data_analysis_variables(df_raw)
load_point_change_times = add_set_points!(df)

println("Cutting the data into dynamic series...")
series = cut_dynamic_series(df, load_point_change_times)

series_dir = joinpath(DATA_DIR, "pilot_plant_series")
mkpath(series_dir)
for (i, df_series) in enumerate(series)
    CSV.write(joinpath(series_dir, "series_$(i).csv"), df_series)
    @printf("  Series %2d: %s to %s (%.1f h, %d samples)\n", i, df_series.time[1], df_series.time[end],
            (df_series.t[end] - df_series.t[1]), nrow(df_series))
end
println("Saved data/pilot_plant_series/series_1.csv ... series_$(length(series)).csv")
