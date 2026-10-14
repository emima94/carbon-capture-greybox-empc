# ------------------------------------------------------------------------------------------
# Core setup for all reproduction drivers: packages, source files, paths and helpers.
# (Without plotting packages, so it also runs in the estimation environment, estimation_env/.)
# ------------------------------------------------------------------------------------------

using LinearAlgebra
using Statistics
using Printf
using Dates
using ComponentArrays
using ForwardDiff
using NLsolve
using CSV, DataFrames
using JLD2, JSON3

# Model, EKF, data and plotting functions (verbatim copies of the original source files)
# (the plotting functions are only defined here; they need the packages loaded in setup.jl)
include("util_diffeq_sys_id.jl")
include("util_diffeq_sys_id_plot.jl")
include("util_model_def.jl")
include("util_general.jl")
include("ekf.jl")
include("models/model_def.jl")
include("models/evaluate_Na.jl")
include("models/system_eq.jl")

# Paths
const REPRO_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(REPRO_DIR, "data")
const RESULTS_DIR = joinpath(REPRO_DIR, "results")
const FIG_DIR = joinpath(REPRO_DIR, "figures")
mkpath(RESULTS_DIR)
mkpath(FIG_DIR)

# NN regularization values, lambda_reg = 10^(-4.0:0.5:1.0), and the value chosen in the paper
const LOG10_LAMBDA_VALUES = collect(-4.0:0.5:1.0)
const LOG10_LAMBDA_CHOSEN = -1.0

# EKF j-step-ahead prediction horizons [h]
const PREDICTION_HORIZONS = [10.0/60.0, 20.0/60.0, 30.0/60.0, 1.0, 2.0, 3.0, 4.0, 6.0]

# Initial state guess [ca, cd, Tld_HX] used to find steady states (from initialize_parameters in util_model_def.jl)
const X0_GUESS = [2.63; 2.10; 90]

"""
    load_pilot_plant_data()

Load the 10 DTI pilot-plant data series (campaign 4, Nov-Dec 2025), downsampled to a 2 min
sampling time. Returns a vector of named tuples with fields t [h], Y (outputs), u (inputs),
dist (disturbances), misc (capture rate), series, etc.
"""
function load_pilot_plant_data()
    datasets, df = load_data_sys_id_DTI(; every_k = 6, data_folder_path = joinpath(DATA_DIR, "pilot_plant_series"))
    @assert all(ds.camp == 4 for ds in datasets)
    return datasets, df
end

"""
    build_dti_model(lambda_reg)

Build the NN-based grey-box model of the DTI pilot plant used in the paper (model "M1" of
config_025 for campaign 4): states [ca, cd, Tld_HX], NNs with 4 neurons for the desorption
rate, Nd(cd, Tld_HX, F), and the reboiler duty, Q(cd, Tld_HX, F, Tld_HX_SP).
Returns the system equations, `ffun_bar(x, u, d, p, p_fixed)`, the output function,
`hfun(x, u, d, p, p_fixed)`, and the fixed parameters, `p_fixed`.
"""
function build_dti_model(lambda_reg)
    NN_def = (Nd = (input_names = [:cd, :Tld_HX, :F], input_func = Nd_input_func, n_neurons = [4]),
              Q = (input_names = [:cd, :Tld_HX, :F, :Tld_HX_SP], input_func = Q_input_func_cd_Tld_F_TlD_SP, n_neurons = [4]))
    plant, w_Ta, with_Q = "DTI", false, true
    _, ffun_bar, hfun, _, p_fixed, _, _, _, _ = get_model(nothing, NN_def, plant, w_Ta, with_Q,
                                                          evaluate_Na_ca_cga_Ta, get_normalization_params_high_mu;
                                                          lambda_reg = lambda_reg)
    return (ffun_bar = ffun_bar, hfun = hfun, p_fixed = p_fixed)
end

# Convert parameter estimates (views/ComponentArrays) to plain arrays, so they load with JLD2 only
to_plain(x::NamedTuple) = NamedTuple{keys(x)}(map(to_plain, values(x)))
to_plain(x::AbstractArray{<:Number}) = Array{Float64}(x)
to_plain(x::AbstractVector) = [to_plain(xi) for xi in x]
to_plain(x::Number) = Float64(x)

lambda_folder(log10_lambda) = @sprintf("log10_lambda_%+.1f", log10_lambda)

"""
    load_parameter_estimates(log10_lambda)

Load the in-sample and the 10 leave-one-out parameter estimates for a given NN regularization
value. Returns a vector of named tuples (sample_type, left_out_series, p, x0, train_series).
"""
function load_parameter_estimates(log10_lambda)
    index = CSV.read(joinpath(DATA_DIR, "parameter_estimates", "index.csv"), DataFrame)
    rows = index[index.log10_lambda_reg .== log10_lambda, :]
    estimates = map(eachrow(rows)) do row
        f = jldopen(joinpath(DATA_DIR, "parameter_estimates", row.folder, row.file))
        est = (sample_type = String(row.sample_type), left_out_series = row.left_out_series,
               lambda_reg = f["lambda_reg"], p = f["p"], x0 = f["x0"], train_series = f["train_series"])
        close(f)
        est
    end
    return estimates
end
