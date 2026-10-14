# ------------------------------------------------------------------------------------------
# Closed-loop EMPC of the DTI pilot plant (Section 7), replayed on the logged plant data
#
# During the closed-loop tests #3 (baseline) and #4 (flexible operation), the EMPC ran in real
# time: every 10 min the controller loaded the newest plant data, updated the state estimate
# with the EKF, solved the OCP with IPOPT and sent the first optimal inputs to the plant.
# This driver runs the same controller offline on the logged data of the two tests. In each
# control interval it uses the plant data up to the start of the control interval, and
#   1) accumulates the captured and inlet CO2 since the start of the test,
#   2) updates the state estimate with the EKF using the data of the last control interval,
#   3) sets up the OCP with the remaining capture requirement, the current disturbance
#      estimate (30 min mean) and the energy price over the horizon, and solves it with IPOPT,
#   4) applies the first optimal inputs in the next control interval.
# The inputs computed offline are compared with the inputs computed by the controller during
# the tests (controller_outputs.json).
#
# Output: results/empc_closed_loop_test_{3,4}.csv, results/empc_closed_loop_test_{3,4}.jld2,
#         figures/empc_closed_loop_replay.pdf
# Runtime: a few minutes
# ------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
using JuMP, Ipopt
import MathOptInterface as MOI
using TimeZones     # Needed to parse the time zone in the time stamps of the measurements
include(joinpath(@__DIR__, "..", "src", "closed_loop_data.jl"))
include(joinpath(@__DIR__, "..", "src", "models_JuMP.jl"))
include(joinpath(@__DIR__, "..", "src", "empc_settings.jl"))
set_plot_defaults(fontfamily = "times", titlefontsize = 9, guidefontsize = 9)

tests_dir = joinpath(DATA_DIR, "closed_loop_tests")
tests = CSV.read(joinpath(tests_dir, "tests.csv"), DataFrame)
closed_loop_tests = tests[tests.loop .== "closed", :]

"Initial controller state: steady state at the initial inputs (initialize_closed_loop_controller_state)."
function initial_controller_state(settings)
    P = 0.1 * I(settings.nx)
    t_vec = 0:10/3600:2
    u_sim_ss = repeat(settings.u_prev, inner = (1, length(t_vec)))
    d_sim_ss = repeat(settings.dist_base, inner = (1, length(t_vec)))
    X = simulate_system(settings.funcs.ffun_2du, settings.x0_opt[1], u_sim_ss, d_sim_ss, settings.p_opt, settings.p_fixed, t_vec)
    sol = nlsolve(x -> settings.funcs.ffun_2du(x, settings.u_prev, settings.dist_base, settings.p_opt, settings.p_fixed), X[:, end])
    return (x_estimate = sol.zero, P = P, Mga = 0.0, Mgina = 0.0, u_prev = settings.u_prev,
            time_prev = settings.time_start, time_current = settings.time_start)
end

"""
One control interval of the closed-loop EMPC (run_closed_loop_iteration_RT in the original
project), with the controller state passed in memory instead of a json file.
"""
function closed_loop_iteration(state, k, df, settings)
    N, Nx = settings.N, settings.Nx
    p_opt, p_fixed = settings.p_opt, settings.p_fixed
    time_prev, time_current = state.time_prev, state.time_current

    # Data of the last 30 min and of the last control interval
    df_30 = df[df.time .>= time_current - Minute(30) .&& df.time .<= time_current, :]
    df_current_interval, t_current_interval = get_control_interval_rows(df_30, time_prev, time_current)

    # Accumulated captured and inlet CO2 since the start of the test
    Mga, Mgina = update_CO2_accumulated!(state.Mga, state.Mgina, df_current_interval, t_current_interval)

    # EKF state estimation over the last control interval
    dist, y_meas = get_dist_and_meas_val_current_interval(df_current_interval)
    dist_ekf = hcat(dist, dist[:, end])
    y_meas_ekf = hcat(y_meas, y_meas[:, end])
    u_ekf = repeat(state.u_prev, 1, length(t_current_interval))
    _, x_post, _, _, _, P_post, _, _ = ekf_filter(settings.funcs.ffun_ekf, settings.funcs.hfun_ekf, state.x_estimate, state.P,
                                                  u_ekf, dist_ekf, p_opt, settings.p_fixed_ekf, t_current_interval, y_meas_ekf)
    x_estimate = x_post[:, end]
    P = P_post[end]

    # Optimal control problem. The capture requirement applies at the end of the test (horizon 1)
    # and over the following 6 h (horizon 2).
    control_interval_start_horizon2 = N - k + 2
    horizon_idx = (0:N-1) .+ k
    D_current = mean(Matrix(df_30[:, [:cgina, :Fgina]])', dims = 2)    # Disturbance estimate: 30 min mean

    p_scenario = (
        p_el = repeat(settings.p_el[horizon_idx], inner = (Int((Nx-1)/N), 1)),
        lambda = settings.lambda,
        eta_min = settings.eta_min,
        x0 = [x_estimate; Mga; Mgina],
        lb = settings.lb,
        ub = settings.ub,
        N = N,
        Nx = Nx,
        dt = settings.dt_control_interval,
        dt_sim = settings.dt_sim,
        ffun = settings.funcs.ffun_JuMP,
        hfun_Q = settings.funcs.hfun_Q_JuMP,
        hfun_cap_eff = settings.funcs.hfun_cap_eff_JuMP,
        u_prev = state.u_prev,
        dist = repeat(D_current, inner = (1, Nx)),
        du = settings.du,
        control_interval_start_horizon2 = control_interval_start_horizon2
    )

    model = settings.funcs.OCP_model_func(p_scenario, p_opt, p_fixed)
    set_silent(model)
    optimize!(model)
    u_opt = collect(value.(model[:u]))

    result = (termination_status = string(termination_status(model)), objective = objective_value(model),
              solve_time = solve_time(model), u_opt = u_opt, x_estimate = x_estimate, Mga = Mga, Mgina = Mgina,
              D_current = vec(D_current))
    new_state = (x_estimate = x_estimate, P = P, Mga = Mga, Mgina = Mgina, u_prev = u_opt[:, 1],
                 time_prev = time_current, time_current = time_current + Minute(10))
    return new_state, result
end

replays = Dict{Int, Any}()
for test in eachrow(closed_loop_tests)
    println("\nTest #$(test.test): $(test.scenario), closed-loop")
    test_dir = joinpath(tests_dir, test.folder)
    time_start = DateTime(test.time_start)
    settings = empc_settings(test.scenario, time_start)

    # Raw plant data as received by the controller, and the controller log of the test. For test #4,
    # the data received online (controller_measurements.csv) differs slightly from the plant
    # historian export used in the evaluation (measurements.csv).
    data_file = isfile(joinpath(test_dir, "controller_measurements.csv")) ? "controller_measurements.csv" : "measurements.csv"
    println("Plant data: $(data_file)")
    df_long = CSV.read(joinpath(test_dir, data_file), DataFrame, dateformat = "yyyy-mm-dd HH:MM:SS.zzz")
    df_long[!, :ts] = String.(df_long[!, :ts])
    ts_long = parse_zdt.(df_long.ts)
    logged_u = JSON3.read(read(joinpath(test_dir, "controller_outputs.json"), String))
    logged_stats = JSON3.read(read(joinpath(test_dir, "solver_stats.json"), String))

    state = initial_controller_state(settings)
    summary = DataFrame(interval = Int[], time = DateTime[], status = String[], solve_time_s = Float64[],
                        objective = Float64[], objective_logged = Float64[],
                        F = Float64[], F_logged = Float64[], Td_sp = Float64[], Td_sp_logged = Float64[],
                        max_diff_u_trajectory = Float64[])
    results = []
    for k in 1:settings.N
        # Plant data available to the controller: data up to the start of the control interval,
        # processed as by the controller during the tests
        @assert DateTime(logged_stats[k]["timestamp_current"]) == state.time_current
        df = process_data(choose_tags_and_reshape_long_df_controller(df_long[ts_long .<= state.time_current, :]))

        state, result = closed_loop_iteration(state, k, df, settings)
        push!(results, result)

        u_logged = hcat(Float64.(logged_u[k]["u_opt"][1]), Float64.(logged_u[k]["u_opt"][2]))'
        push!(summary, (k, state.time_prev, result.termination_status, result.solve_time, result.objective,
                        logged_stats[k]["objective_value"], result.u_opt[1, 1], u_logged[1, 1],
                        result.u_opt[2, 1], u_logged[2, 1], maximum(abs.(result.u_opt .- u_logged))))
        @printf("  Interval %2d (%s): %s, F = %.4f (logged %.4f), Td_sp = %.3f (logged %.3f)\n", k,
                Dates.format(state.time_prev, "HH:MM"), result.termination_status, result.u_opt[1, 1], u_logged[1, 1],
                result.u_opt[2, 1], u_logged[2, 1])
    end

    CSV.write(joinpath(RESULTS_DIR, "empc_closed_loop_test_$(test.test).csv"), summary)
    jldsave(joinpath(RESULTS_DIR, "empc_closed_loop_test_$(test.test).jld2"); results)
    replays[test.test] = summary

    @printf("Test #%d: max. abs. difference to the logged first inputs: F %.2e m3/h, Td_sp %.2e °C\n", test.test,
            maximum(abs.(summary.F .- summary.F_logged)), maximum(abs.(summary.Td_sp .- summary.Td_sp_logged)))
end

## Figure: inputs computed offline vs. inputs computed by the controller during the tests
t_control(n) = collect(0:n) .* (10 / 60)
plts = []
for (test_no, color) in [(3, colors_palette.blue), (4, colors_palette.orange)]
    s = replays[test_no]
    for (var, var_logged, ylbl) in [(:F, :F_logged, "F [m³/h]"), (:Td_sp, :Td_sp_logged, "Td_sp [°C]")]
        plt = plot(t_control(nrow(s)), [s[!, var_logged]; s[end, var_logged]], seriestype = :steppost, lw = 3, color = color,
                   label = "Test #$(test_no), online", xlabel = "Time [h]", ylabel = ylbl)
        plot!(plt, t_control(nrow(s)), [s[!, var]; s[end, var]], seriestype = :steppost, lw = 1.5, color = :black,
              linestyle = :dash, label = "Test #$(test_no), offline replay")
        push!(plts, plt)
    end
end
savefig(plot(plts..., layout = (2, 2), size = (800, 500), left_margin = 3mm), joinpath(FIG_DIR, "empc_closed_loop_replay.pdf"))
println("\nSaved results/empc_closed_loop_test_{3,4}.csv and figures/empc_closed_loop_replay.pdf")
