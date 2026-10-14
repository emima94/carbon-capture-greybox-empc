# ------------------------------------------------------------------------------------------
# Compute EKF-based j-step-ahead predictions and RMSEs (Section 6.1)
#
# For every NN regularization value, lambda_reg, and every parameter estimate (in-sample and
# 10 leave-one-out estimates), we
#   1) filter the data series with the EKF to get the filtered state estimates x_{k|k},
#   2) predict j steps ahead from x_{k|k} for each prediction horizon,
#   3) compute the RMSE of the predictions of y_g and Q.
# In-sample estimates are evaluated on all 10 series, using their estimated initial states.
# Leave-one-out estimates are evaluated on the left-out series only, starting from the steady
# state at the first sample.
#
# Output: results/predictions.jld2 (used by drivers 05 and 06) and results/rmse_summary.csv
# Runtime: a few minutes with multiple threads (julia --project=. -t auto ...)
# ------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "src", "setup.jl"))

datasets, _ = load_pilot_plant_data()

# EKF settings (get_EKF_noise_params("DTI", "camp_4", ...) in src/ekf.jl)
nx = 3                                  # States: [ca, cd, Tld_HX]
ny = 3                                  # Observations: [yg, Tld_HX, Q]
sx, Rv, _, _ = get_EKF_noise_params("DTI", "camp_4", nx, ny)
P0 = Diagonal(fill(0.01, nx))           # Initial state covariance
dt_sim_euler = 10 / 3600                # Euler step in the EKF prediction [h]

# All (lambda_reg, parameter estimate) combinations
tasks = [(log10_lambda = l, est = est) for l in LOG10_LAMBDA_VALUES for est in load_parameter_estimates(l)]
println("Computing predictions for $(length(tasks)) parameter estimates using $(Threads.nthreads()) thread(s)...")

predictions_per_task = Vector{Any}(undef, length(tasks))
counter = Threads.Atomic{Int}(0)

Threads.@threads for i in eachindex(tasks)
    log10_lambda, est = tasks[i]
    p = est.p
    model = build_dti_model(est.lambda_reg)
    ffun_bar, hfun, p_fixed = model.ffun_bar, model.hfun, model.p_fixed
    p_fixed_ekf = (; p_fixed..., nx = nx, Rv = Rv, sx = sx, dt = dt_sim_euler)

    ffun_ekf = (x, u, d, p, p_fixed) -> moment_equations_EKF(x, u, d, p, p_fixed, ffun_bar)
    hfun_ekf = (x, u, d, p, p_fixed) -> p_fixed.hfun(x, u, d, p, p_fixed)

    series_to_evaluate = est.sample_type == "in-sample" ? collect(1:length(datasets)) : [est.left_out_series]

    predictions_i = []
    for m in series_to_evaluate
        ds = datasets[m]

        # Initial state
        if est.sample_type == "in-sample"
            x0 = est.x0[m]                     # Estimated initial state of series m
        else
            x0 = get_steady_state(ffun_bar, X0_GUESS, ds.u[:, 1], ds.dist[:, 1], p, p_fixed)
        end

        # Filtered state estimates
        _, x_post, _, _, _, _, _, _ = ekf_filter(ffun_ekf, hfun_ekf, x0, P0, ds.u, ds.dist, p, p_fixed_ekf, ds.t, ds.Y[p_fixed.obs_idx, :])

        # j-step-ahead predictions and RMSEs for each prediction horizon
        yg_pred, Q_pred, RMSE_yg, RMSE_Q = Vector{Float64}[], Vector{Float64}[], Float64[], Float64[]
        for horizon in PREDICTION_HORIZONS
            _, y_pred = ekf_tk_ahead_prediction_general(ffun_bar, hfun, x_post, ds.u, ds.dist, p, p_fixed_ekf, ds.t, horizon)
            y_named = named_output(y_pred)
            push!(yg_pred, Float64.(y_named.yga))
            push!(Q_pred, Float64.(y_named.Q))
            push!(RMSE_yg, sqrt(mean((y_named.yga - ds.Y[1, :]).^2)))
            push!(RMSE_Q, sqrt(mean((y_named.Q - ds.Y[4, :]).^2)))
        end

        push!(predictions_i, (log10_lambda = log10_lambda, lambda_reg = est.lambda_reg,
                              sample_type = est.sample_type, series = m,
                              RMSE_yg = RMSE_yg, RMSE_Q = RMSE_Q,
                              yg_pred = yg_pred, Q_pred = Q_pred))
    end
    predictions_per_task[i] = predictions_i

    Threads.atomic_add!(counter, 1)
    println("  $(counter[])/$(length(tasks)) done")
end

predictions = [pred for preds in predictions_per_task for pred in preds]
prediction_horizons = PREDICTION_HORIZONS
jldsave(joinpath(RESULTS_DIR, "predictions.jld2"); predictions, prediction_horizons)

## Summary table: cross-validated out-of-sample RMSE, phi_oos, and in-sample RMSE, phi_is (Section 6.1)
rms(x) = sqrt(mean(x .^ 2))
summary = DataFrame(log10_lambda_reg = Float64[], horizon_h = Float64[],
                    phi_is_yg = Float64[], phi_oos_yg = Float64[], phi_is_Q = Float64[], phi_oos_Q = Float64[])
for log10_lambda in LOG10_LAMBDA_VALUES, (h, horizon) in enumerate(PREDICTION_HORIZONS)
    is = filter(r -> r.log10_lambda == log10_lambda && r.sample_type == "in-sample", predictions)
    oos = filter(r -> r.log10_lambda == log10_lambda && r.sample_type == "out-of-sample", predictions)
    push!(summary, (log10_lambda, horizon,
                    rms([r.RMSE_yg[h] for r in is]), rms([r.RMSE_yg[h] for r in oos]),
                    rms([r.RMSE_Q[h] for r in is]), rms([r.RMSE_Q[h] for r in oos])))
end
CSV.write(joinpath(RESULTS_DIR, "rmse_summary.csv"), summary)

println("\nCross-validated RMSE, 1 h ahead predictions:")
show(stdout, MIME("text/plain"), summary[summary.horizon_h .== 1.0, :]; allrows = true)
println("\n\nSaved results/predictions.jld2 and results/rmse_summary.csv")
