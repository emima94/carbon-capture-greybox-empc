# ------------------------------------------------------------------------------------------
# Parameter estimation of the NN-based grey-box model (Section 4)
#
# Estimates the model parameters by minimizing the RMSE of the full-horizon predictions of the
# observations (y_g, T_d, Q) plus the NN weight regularization. For each NN regularization
# value lambda_reg there are 11 estimates ("runs"):
#   run 1      in-sample: trained on all 10 series
#   run j > 1  leave-one-out: trained on all series except series j-1
# Optimization: Adam (lr = 0.05, max. 2500 iterations, early stopping), then L-BFGS (max. 5000
# iterations). The NN parameters are initialized randomly and then fitted to steady-state data.
#
# In the paper, all 121 estimates (11 lambda_reg values x 11 runs) were computed on an HPC
# cluster, one process per lambda_reg value (driver/driver_sys_id_hpc.jl, config_025, in the
# original project). One run takes 10-50 min.
#
# The driver runs in its own environment, estimation_env/, with the package versions used on
# the HPC cluster. The random number generator is seeded and advanced as in the HPC processes,
# so the random initial parameters are the same. The NN weights are not uniquely identifiable,
# so tiny floating-point differences between computers (e.g. 1e-9 in the initial loss) can
# lead the optimizer to different weights with practically the same loss. The driver therefore
# compares the final loss with the loss of the published estimate.
#
# Usage:
#     julia --project=estimation_env -e "using Pkg; Pkg.instantiate()"     # once
#     julia --project=estimation_env --threads=auto drivers/02_estimate_parameters.jl <log10_lambda> <run>
#   e.g.
#     ... drivers/02_estimate_parameters.jl -1.0 1      # lambda_reg = 0.1, in-sample
#     ... drivers/02_estimate_parameters.jl -1.0 all    # all 11 runs for lambda_reg = 0.1
#
# Output: results/parameter_estimates/log10_lambda_<value>/{in_sample, oos_series_XX}.jld2 (same
#         format as data/parameter_estimates/), and the losses of the new and the published estimates
# ------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "src", "setup_core.jl"))
using DifferentialEquations
import SciMLSensitivity as SMS
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LineSearches
using DataInterpolations
using Random

log10_lambda = parse(Float64, get(ARGS, 1, string(LOG10_LAMBDA_CHOSEN)))
runs_arg = get(ARGS, 2, "1")
n_series = 10
n_runs = n_series + 1
runs = runs_arg == "all" ? collect(1:n_runs) : [parse(Int, runs_arg)]
@assert log10_lambda in LOG10_LAMBDA_VALUES "log10_lambda must be one of $(LOG10_LAMBDA_VALUES)"
@assert all(1 .<= runs .<= n_runs) "run must be between 1 and $(n_runs), or all"

# lambda_reg exactly as in the original configuration, 10 .^ (-4:0.5:1)
lambda_reg = (10 .^ (-4:0.5:1))[findfirst(==(log10_lambda), LOG10_LAMBDA_VALUES)]

datasets, _ = load_pilot_plant_data()
train_series(run) = run == 1 ? collect(1:n_series) : setdiff(1:n_series, run - 1)

# Model definition (as in build_dti_model in src/setup.jl). With datasets given, get_model
# also initializes the NN parameters by fitting them to steady-state data.
NN_def = (Nd = (input_names = [:cd, :Tld_HX, :F], input_func = Nd_input_func, n_neurons = [4]),
          Q = (input_names = [:cd, :Tld_HX, :F, :Tld_HX_SP], input_func = Q_input_func_cd_Tld_F_TlD_SP, n_neurons = [4]))
model(ds) = get_model(ds, NN_def, "DTI", false, true, evaluate_Na_ca_cga_Ta, get_normalization_params_high_mu; lambda_reg = lambda_reg)
sensealg = SMS.InterpolatingAdjoint(autojacvec = SMS.ZygoteVJP())

# Random initial NN parameters: in the HPC process, the generator was seeded once, the model
# was set up once for a test of the loss function, and then the 11 runs were computed in
# order, each drawing new initial NN weights. Skipped runs only advance the generator.
NN_settings = model(nothing)[end]   # (draws random numbers, so it is called before seeding)
Random.seed!(1234)
advance_rng() = (initialize_NN_params_from_settings(NN_settings[:Nd]); initialize_NN_params_from_settings(NN_settings[:Q]))
advance_rng()  # Test of the loss function in the HPC process

out_dir = joinpath(RESULTS_DIR, "parameter_estimates", lambda_folder(log10_lambda))
mkpath(out_dir)
published = Dict((e.sample_type == "in-sample" ? 1 : e.left_out_series + 1) => e for e in load_parameter_estimates(log10_lambda))

flatten(p::NamedTuple) = vcat([flatten(v) for v in values(p)]...)
flatten(x::AbstractArray) = vec(Float64.(x))
flatten(x::Number) = [Float64(x)]

for run in 1:n_runs
    if !(run in runs)
        advance_rng()
        continue
    end

    ds = datasets[train_series(run)]
    file = run == 1 ? "in_sample.jld2" : @sprintf("oos_series_%02d.jld2", run - 1)
    println("\n", "="^90)
    println("lambda_reg = $(lambda_reg), run $(run)/$(n_runs): training series $(train_series(run))")
    println("Started $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

    # Model and initial parameters
    _, _, _, θ0, p_fixed, _, _, prob_list, _ = model(ds)
    loss_fn = (θ, _) -> loss(θ, ds, prob_list, sensealg, p_fixed)
    initial_loss = loss_fn(θ0, nothing)
    println("Initial loss: $(initial_loss)")

    # Optimization (Adam + L-BFGS)
    global loss_history = Float64[]
    p_opt, x0_opt, _, loss_history_run, _, _ = optimize_2_step(loss_fn, θ0)

    # Save in the same format as the published estimates
    p = to_plain(p_opt)
    x0 = [Float64.(x) for x in x0_opt]
    jldsave(joinpath(out_dir, file); p, x0, train_series = train_series(run), lambda_reg, initial_loss,
            loss_history = loss_history_run)

    # Compare with the published estimate
    θ_pub = ComponentArray((p = published[run].p, x0 = published[run].x0))
    loss_new = loss_fn(ComponentArray((p = p, x0 = x0)), nothing)
    loss_pub = loss_fn(θ_pub, nothing)
    max_diff = maximum(abs.(flatten(p) .- flatten(published[run].p)))
    println("Finished $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")).")
    @printf("Loss of the new estimate: %.6f, loss of the published estimate (data/parameter_estimates/%s/%s): %.6f\n",
            loss_new, lambda_folder(log10_lambda), file, loss_pub)
    @printf("Max. abs. difference between the new and the published parameters: %.3e\n", max_diff)
end
