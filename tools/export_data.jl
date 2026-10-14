# ------------------------------------------------------------------------------------------
# One-time export of the data needed for reproduction into `reproduction/data/` (in the original
# CarbonCaptureMPC project, this repository was the folder `reproduction/`).
#
# This script is NOT needed to reproduce the results. It documents how the published data
# files were created from the original (unpublished) project folders. It must be run from
# the root of the original CarbonCaptureMPC project, in the original project environment,
# since loading the original parameter files requires the optimization packages:
#
#     julia --project=. reproduction/tools/export_data.jl
#
# It writes:
#   data/raw_pilot_plant/         The raw logs of the pilot plant from measurement campaign 4
#                                 (Nov-Dec 2025), from the DTI_pilot_data_analysis folder. The data
#                                 series in data/pilot_plant_series/ are computed from these by
#                                 drivers/01_process_raw_data.jl.
#   data/parameter_estimates/     Parameter estimates (config_025, campaign 4, nonlinear models)
#                                 converted to plain Julia types
#   data/closed_loop_tests/       Measurements, applied inputs, controller outputs and solver
#                                 statistics from the four EMPC tests (April 2026)
# ------------------------------------------------------------------------------------------

using Pkg
Pkg.activate(".")

using DifferentialEquations
import SciMLSensitivity as SMS
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LineSearches, ForwardDiff, DataInterpolations, LinearAlgebra, ComponentArrays, NLsolve
using Statistics, Printf, Dates
using CSV, DataFrames, JLD2, JSON3
using DuckDB

include("../../src/util_diffeq_sys_id.jl")
include("../../src/util_model_def.jl")
include("../../src/util_general.jl")
include("../../src/util_diffeq_sys_id_plot.jl")
include("../../src/ekf.jl")
include("../../src/models/model_def.jl")
include("../../src/models/evaluate_Na.jl")
include("../../src/models/system_eq.jl")
include("../../src/configs/config_020.jl")
include("../../src/configs/config_025.jl")
include("../../src/settings/settings_test_260420_01.jl")
include("../../src/settings/settings_test_260420_02.jl")

root_path = pwd()
out_path = joinpath(root_path, "reproduction", "data")
mkpath(out_path)

## ------------------------------------------------------------------------------------------
# 1) Raw pilot-plant logs of measurement campaign 4 (Nov-Dec 2025)
# ------------------------------------------------------------------------------------------
# The paper uses the data series of campaign 4. In the original data analysis
# (DTI_pilot_data_analysis/driver/driver_ss_analysis.jl) the raw logs of all four campaigns were
# processed into 23 data series, of which series 14-23 are from campaign 4. These are
# reproduced exactly from the campaign-4 logs by drivers/01_process_raw_data.jl.
raw_src_folder = joinpath(dirname(root_path), "DTI_pilot_data_analysis", "data")
raw_dst_folder = joinpath(out_path, "raw_pilot_plant")
mkpath(raw_dst_folder)
raw_files = ["111 - 2025-11-24 Default.txt", "112 - 2025-11-27 Default.txt", "113 - 2025-11-27 Default.txt",
             "116 - 2025-12-01 Default.txt", "117 - 2025-12-03 Default.txt"]
for file in raw_files
    cp(joinpath(raw_src_folder, file), joinpath(raw_dst_folder, file); force = true)
end
println("Copied $(length(raw_files)) raw log files.")

# The processed data series of the original project (used below for the model configuration)
src_series_folder = joinpath(root_path, "data", "DTI_23_series")
datasets_all_DTI, _ = load_data_sys_id_DTI(; every_k = 6, data_folder_path = src_series_folder)
@assert findall(ds -> ds.camp == 4, datasets_all_DTI) == collect(14:23)

## ------------------------------------------------------------------------------------------
# 2) Parameter estimates (computed on HPC with driver/driver_sys_id_hpc.jl, config_025)
# ------------------------------------------------------------------------------------------
# config_025 holds 36 models: 3 campaign groups x (1 linear + 11 nonlinear models). The paper
# uses the nonlinear models for campaign 4, i.e. model 14-24, one for each NN regularization
# value lambda_reg = 10^(-4.0:0.5:1.0). Each model has 11 parameter estimates:
# run 1 = in-sample (trained on all 10 series), run j = 2..11 = leave-one-out (series j-1 left out).
datasets_all_Tiller, _ = load_data_sys_id_Tiller(; every_k = 6)
camp123_with_Q_idxs = [ds.camp in [1, 2, 3] for ds in datasets_all_DTI]
camp123_with_Q_idxs[[1, 2, 3]] .= false
camp4_idxs = [ds.camp == 4 for ds in datasets_all_DTI]
camp_Tiller_idxs = [true for _ in datasets_all_Tiller]
_, _, config_name, _, config = config_025((datasets_all_DTI, datasets_all_Tiller),
                                          (camp123_with_Q_idxs, camp4_idxs, camp_Tiller_idxs))

model_idxs_camp4_nonlinear = 14:24
log10_lambda_values = collect(-4.0:0.5:1.0)

# Convert views/ComponentArrays to plain arrays so the files load with JLD2 only
to_plain(x::NamedTuple) = NamedTuple{keys(x)}(map(to_plain, values(x)))
to_plain(x::AbstractArray{<:Number}) = Array{Float64}(x)
to_plain(x::AbstractVector) = [to_plain(xi) for xi in x]
to_plain(x::Number) = Float64(x)

par_folder = joinpath(out_path, "parameter_estimates")
mkpath(par_folder)
index_df = DataFrame(folder = String[], file = String[], log10_lambda_reg = Float64[], lambda_reg = Float64[],
                     sample_type = String[], train_series = String[], left_out_series = Int[], original_file = String[])

for (model_idx, log10_lambda) in zip(model_idxs_camp4_nonlinear, log10_lambda_values)
    # Check that the model is the expected one
    @assert config.plant_list[model_idx] == "DTI"
    @assert config.camp_idxs_list[model_idx] == camp4_idxs
    @assert config.NN_config.Nd_n_neurons_list[model_idx] == [4]
    @assert config.NN_config.Q_n_neurons_list[model_idx] == [4]
    @assert config.with_Q_list[model_idx] && !config.w_Ta_list[model_idx]
    @assert config.lambda_reg_list[model_idx] ≈ 10^log10_lambda

    folder = @sprintf("log10_lambda_%+.1f", log10_lambda)
    mkpath(joinpath(par_folder, folder))
    n_runs = length(config.train_idxs_list_list[model_idx])
    for j in 1:n_runs
        original_file = "results/sys_id/$(config_name)/M$(model_idx)/$(j)-$(n_runs).jld2"
        @load original_file res_named_tuple
        @assert res_named_tuple.train_idxs == config.train_idxs_list_list[model_idx][j]

        train_series = collect(res_named_tuple.train_idxs)
        left_out = setdiff(1:10, train_series)
        if isempty(left_out)
            sample_type, left_out_series, file = "in-sample", 0, "in_sample.jld2"
        else
            sample_type, left_out_series = "out-of-sample", only(left_out)
            file = @sprintf("oos_series_%02d.jld2", left_out_series)
        end

        p = to_plain(res_named_tuple.p_opt)
        x0 = [Float64.(x) for x in res_named_tuple.x0_opt]  # Estimated initial states of the training series
        lambda_reg = Float64(config.lambda_reg_list[model_idx])
        jldsave(joinpath(par_folder, folder, file); p, x0, train_series, lambda_reg, original_file)

        push!(index_df, (folder, file, log10_lambda, lambda_reg, sample_type, join(train_series, " "), left_out_series, original_file))
    end
end
CSV.write(joinpath(par_folder, "index.csv"), index_df)
println("Exported $(nrow(index_df)) parameter estimates.")

## ------------------------------------------------------------------------------------------
# 3) EMPC demonstration at the DTI pilot plant (April 2026)
# ------------------------------------------------------------------------------------------
# Test numbering follows the paper:
#   #1 baseline, open-loop    #2 flexible operation, open-loop
#   #3 baseline, closed-loop  #4 flexible operation, closed-loop
# For each test we export
#   measurements.csv   raw historian data (long format) for the sensors used in the evaluation,
#                      from the start of the logged file to 1 h after the 6 h test window.
#                      (The start of the file is kept, since process_data computes the time
#                      relative to the first sample, which affects the floating point rounding.)
#   inputs.csv         applied set points for the solvent flow rate and desorber temperature
#   controller_outputs.json, solver_stats.json  (closed-loop tests only, as logged by the controller)
cl_folder = joinpath(root_path, "data", "closed_loop_testing")
tests = [
    (name = "test_1_baseline_open_loop", csv = "data_260420_02.csv", time_start = DateTime(2026, 4, 20, 18, 0),
     settings_func = get_settings_test_260420_02, output_json = nothing, solver_json = nothing),
    (name = "test_2_flexible_open_loop", csv = "data_260420_01.csv", time_start = DateTime(2026, 4, 20, 10, 0),
     settings_func = get_settings_test_260420_01, output_json = nothing, solver_json = nothing),
    (name = "test_3_baseline_closed_loop", csv = "data_260416.csv", time_start = DateTime(2026, 4, 16, 8, 50),
     settings_func = nothing, output_json = "output_260416.json", solver_json = "solver_stats_2026-04-19_222517.json"),
    (name = "test_4_flexible_closed_loop", csv = "data_260415.csv", time_start = DateTime(2026, 4, 15, 9, 0),
     settings_func = nothing, output_json = "output_2026-04-15_151058.json", solver_json = "solver_stats_2026-04-15_151058.json"),
]

# Sensor tags used in choose_tags_and_reshape_long_df (src/util_closed_loop_MPC.jl)
tags_used = ["CO2_abs_in - [%]", "Flow_før abs - [kg/h]", "CO2_tør_abs - [%]", "A010_BT02 - [°C]",
             "A020_BT01 - [°C]", "A020_BT02 - [°C]", "A060_BT02 - [°C]", "Damp_kedel - [kWh]",
             "A050_BF01 - [m3/h]", "Flue_inlet_temp - [°C]", "Røg - [°C]", "Effekt - [kW]"]

test_duration = Hour(6)
margin = Hour(1)
dt_control_interval = 10.0 / 60.0

for test in tests
    test_folder = joinpath(out_path, "closed_loop_tests", test.name)
    mkpath(test_folder)

    # Measurements: keep the original lines (byte-identical) for the used tags up to 1 h after the test
    t_hi = Dates.format(test.time_start + test_duration + margin, "yyyy-mm-dd HH:MM:SS")
    lines = readlines(joinpath(cl_folder, "historic_csv_data", test.csv))
    header, body = lines[1], lines[2:end]
    @assert header == "ts,sensor,unit,value"
    kept = String[]
    for line in body
        fields = split(line, ',')
        @assert length(fields) == 4
        @assert endswith(fields[1], "+02:00")
        ts = fields[1][1:19]
        if fields[2] in tags_used && ts <= t_hi
            push!(kept, line)
        end
    end
    open(joinpath(test_folder, "measurements.csv"), "w") do io
        println(io, header)
        foreach(l -> println(io, l), kept)
    end

    # Applied inputs
    if test.output_json === nothing  # Open-loop: inputs defined in the test settings
        settings = test.settings_func(root_path)
        u = settings.u
        time_control = Float64.(settings.time_control)
    else                             # Closed-loop: first input of each computed input trajectory
        data_u = JSON3.read(read(joinpath(cl_folder, "output_json_old", test.output_json), String))
        u = [[data_u[i]["u_opt"][1][1], data_u[i]["u_opt"][2][1]] for i in eachindex(data_u)]
        time_control = collect(0.0 .+ (0:length(u)-1) .* dt_control_interval)
        cp(joinpath(cl_folder, "output_json_old", test.output_json), joinpath(test_folder, "controller_outputs.json"); force = true)
        cp(joinpath(cl_folder, "solver_stats_json_old", test.solver_json), joinpath(test_folder, "solver_stats.json"); force = true)
    end
    CSV.write(joinpath(test_folder, "inputs.csv"),
              DataFrame(time_h = time_control, F_sp = [ui[1] for ui in u], Td_sp = [ui[2] for ui in u]))

    println("Exported $(test.name): $(length(kept)) measurement rows, $(length(u)) inputs.")
end

# Plant data received online by the controller in test #4. During the closed-loop tests, the
# controller read the plant data from a DuckDB database that was filled online. For test #3
# that database is identical to the historian export in measurements.csv. For test #4 the
# historian export has a different sampling, so the online data is exported as well (in the
# same format), for the offline replay of the controller (drivers/09_empc_closed_loop.jl).
controller_tags = tags_used[1:9]   # The tags loaded by the controller during the tests
con = DBInterface.connect(DuckDB.DB, joinpath(cl_folder, "historic_db", "db_260415_01.duckdb"); readonly = true)
DBInterface.execute(con, "SET TimeZone = 'Europe/Copenhagen'")
df_online = DataFrame(DBInterface.execute(con, """
    SELECT strftime(ts, '%Y-%m-%d %H:%M:%S.%f') || '+02:00' AS ts, sensor, unit, value FROM readings
    WHERE sensor IN ($(join(["'" * t * "'" for t in controller_tags], ", "))) ORDER BY ts, sensor"""))
DBInterface.close!(con)
df_online.ts = replace.(df_online.ts, ".000000+02:00" => "+02:00")   # Same time format as the historian export
CSV.write(joinpath(out_path, "closed_loop_tests", "test_4_flexible_closed_loop", "controller_measurements.csv"), df_online)
println("Exported the online plant data of test #4: $(nrow(df_online)) rows.")

# Test start times
CSV.write(joinpath(out_path, "closed_loop_tests", "tests.csv"),
          DataFrame(test = 1:4, folder = [t.name for t in tests],
                    time_start = [Dates.format(t.time_start, "yyyy-mm-ddTHH:MM:SS") for t in tests],
                    duration_h = 6.0,
                    loop = ["open", "open", "closed", "closed"],
                    scenario = ["baseline", "flexible", "baseline", "flexible"]))

println("Done.")
