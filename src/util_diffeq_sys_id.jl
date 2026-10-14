
# ============================================================
# Generate ODE problems for each dataset 
# ============================================================

function make_ode_problem(ffun, dataset, x0, p, p_fixed)
    function ode!(dx, x, _p, t)
        u = ConstantInterpolation(dataset.u, dataset.t; extrapolation = ExtrapolationType.Constant)
        d = ConstantInterpolation(dataset.dist, dataset.t; extrapolation = ExtrapolationType.Constant)
        dx .= ffun(x, u, d, _p, t, p_fixed)
    end

    return ODEProblem(ode!, x0, (dataset.t[1], dataset.t[end]), p)
end

function make_dae_problem(ffun, dataset, x0, p, p_fixed)
    function dae!(dx, x, _p, t)
        u = ConstantInterpolation(dataset.u, dataset.t; extrapolation = ExtrapolationType.Constant)
        d = ConstantInterpolation(dataset.dist, dataset.t; extrapolation = ExtrapolationType.Constant)
        dx .= ffun(x, u, d, _p, t, p_fixed)
    end

    M = Diagonal([1.0, 1.0, 1.0, 0.0])

    f = ODEFunction(dae!, mass_matrix = M)

    return ODEProblem(f, x0, (dataset.t[1], dataset.t[end]), p)
end

# ============================================================
# Simulate one dataset, return predicted Y (n_y x N)
# ============================================================


function simulate_dataset(prob, u, d, x0, p, t_vec, sensealg, p_fixed)
    hfun = p_fixed.hfun
    ode_solver_str = p_fixed.ode_solver_str
    if ode_solver_str == "Tsit5"
        ode_solver = Tsit5()
    else
        error("Unsupported ODE solver: $ode_solver_str")
    end

    _prob = remake(prob, u0=x0, p=p)
    X  = Array(solve(_prob, ode_solver;
        tstops  = t_vec,
        saveat  = t_vec,
        abstol  = 1e-6,
        reltol  = 1e-6,
        sensealg = sensealg))

    Y_pred = hcat([hfun(X[:,j], u[:,j], d[:,j], p, p_fixed) for j in 1:length(t_vec)]...)
    return Y_pred, X
end

function simulate_dataset_detailed(prob, u, d, x0, p, t_vec, sensealg, p_fixed)

    _prob = remake(prob, u0=x0, p=p)
    X  = Array(solve(_prob, p_fixed.ode_solver;
        tstops  = t_vec,
        saveat  = t_vec,
        abstol  = 1e-6,
        reltol  = 1e-6,
        sensealg = sensealg))

    Z_pred = hcat([hfun_detailed(X[:,j], u[:,j], d[:,j], p, p_fixed) for j in 1:length(t_vec)]...)
    return Z_pred, X
end

# ============================================================
# Loss function
# ============================================================
function loss(theta, datasets, prob_list, sensealg, p_fixed)
    p   = theta.p
    x0 = theta.x0
    NN_settings = p_fixed.NN_settings
    lambda_reg = p_fixed.lambda_reg
    #p_NN = theta[p_fixed.idx_NN]
    obs_weights = p_fixed.obs_weights
    obs_idx = p_fixed.obs_idx

    mse = zeros(eltype(p), length(datasets))

    Threads.@threads for i in eachindex(datasets)
        ds = datasets[i]
        
        Y_pred, _ = simulate_dataset(prob_list[i], ds.u, ds.dist, x0[i], p, ds.t, sensealg, p_fixed)

        error = Y_pred .- ds.Y[obs_idx, :]
        weighted_error = error .* obs_weights

        mse[i] = mean(weighted_error .^ 2)
    end

    # Number of observations pr. dataset
    n_obs = [ds.N for ds in datasets]
    #total_mse = sum(mse .* n_obs) / sum(n_obs) # Weighted average MSE across datasets
    #total_rmse = sqrt(total_mse)
    
    # Loss like in ifac paper:
    total_rmse = sqrt(mean(mse))
    #total_mse = mean(mse)

    # Parameter regularization (L2 norm) on all NN parameters
    #par_reg = p_fixed.lambda_reg * sum(p_NN .^ 2)
    # Parameter regularization (L2 norm) on all NN weight (W) parameters
    par_reg = compute_NN_regularization(p, NN_settings, lambda_reg)

    loss = total_rmse + par_reg

    return loss

end

function loss_MSE(theta, datasets, prob_list, sensealg, p_fixed)
    p   = theta.p
    x0 = theta.x0
    NN_settings = p_fixed.NN_settings
    lambda_reg = p_fixed.lambda_reg
    #p_NN = theta[p_fixed.idx_NN]
    obs_weights = p_fixed.obs_weights
    obs_idx = p_fixed.obs_idx

    mse = zeros(eltype(p), length(datasets))

    Threads.@threads for i in eachindex(datasets)
        ds = datasets[i]
        
        Y_pred, _ = simulate_dataset(prob_list[i], ds.u, ds.dist, x0[i], p, ds.t, sensealg, p_fixed)

        error = Y_pred .- ds.Y[obs_idx, :]
        weighted_error = error .* obs_weights

        mse[i] = mean(weighted_error .^ 2)
    end

    # Number of observations pr. dataset
    n_obs = [ds.N for ds in datasets]
    #total_mse = sum(mse .* n_obs) / sum(n_obs) # Weighted average MSE across datasets
    #total_rmse = sqrt(total_mse)
    
    # Loss like in ifac paper:
    #total_rmse = sqrt(mean(mse))
    total_mse = mean(mse)

    # Parameter regularization (L2 norm) on all NN parameters
    #par_reg = p_fixed.lambda_reg * sum(p_NN .^ 2)
    # Parameter regularization (L2 norm) on all NN weight (W) parameters
    par_reg = compute_NN_regularization(p, NN_settings, lambda_reg)

    loss = total_mse + par_reg

    return loss

end


function compute_NN_regularization(p, NN_settings, lambda_reg)

    """
    This function computes the L2 regularization term for the weights of the neural networks defined in NN_settings. 
        Only if the NN is nonlinear (i.e., has hidden layers). Only the weights, W, are regularized, not the biases.
    """

    loss_reg = zero(eltype(p))
    for nn_name in string.(keys(NN_settings))
        nn_setting = NN_settings[Symbol(nn_name)]
        n_neurons = nn_setting.n_neurons
        if !isempty(n_neurons) # Only include if it's NOT a linear transformation
            n_hidden_layers = length(n_neurons) # number of hidden layers
            for i in 1:n_hidden_layers+1
                W = p[nn_name * "_params"]["layer_$(i)"].W
                #loss_reg += sum(W .^ 2)
                loss_reg += mean(W .^ 2) # Use mean instead of sum to make the regularization term independent of the NN size
            end
            loss_reg *= 1/(n_hidden_layers+1) # Average over the number of layers to make the regularization term independent of the NN size
        end
    end
    loss_reg *= lambda_reg # Scale by the regularization parameter
    return loss_reg
end


    

# callback
function callback_opt(state, l)
    push!(iter_log, (state.iter, l))
    mod(state.iter, 5) == 0 && println("  iter $(state.iter): loss = $(round(l, digits=8))")
    false
end

# Load data Tiller
function load_data_sys_id_Tiller(; every_k = 6)
    df = Vector{DataFrame}(undef, 10)
    t = Vector{Vector{Float64}}(undef, 10)
    series = Vector{Int}(undef, 10)
    Y = Vector{Matrix}(undef, 10)
    U = Vector{Matrix}(undef, 10)
    D = Vector{Matrix}(undef, 10)
    misc = Vector{Matrix}(undef, 10)
    for i in 1:10
        df[i] = CSV.read("data/Tiller_edit_0326/series_$(i).csv", DataFrame)
        t[i] = df[i].t[1:every_k:end]
        Y[i] = hcat(df[i].ygA[1:every_k:end])'
        #U[i] =  hcat(df[i].F_co_A2D_vol, df[i].P_reb)[1:every_k:end,:]'#, df[i].cginA, df[i].FginA))
        U[i] =  hcat(df[i].F_co_D2A_vol, df[i].P_reb)[1:every_k:end,:]'#, df[i].cginA, df[i].FginA))
        D[i] = hcat(df[i].cginA, df[i].FginA)[1:every_k:end,:]'
        misc[i] = hcat(df[i].N_A)[1:every_k:end]'
        series[i] = i
    end

    camp = fill(1, 10) # All datasets from the same campaign
    time = fill(0, 10)

    datasets = [ (t = t[i], time = time[i], Y = Y[i], u = U[i], dist = D[i], misc = misc[i], N = length(t[i]), camp = camp[i], series = series[i]) for i in 1:10 ]
    return datasets, df
end


# Load data 
function load_data_sys_id_DTI(; every_k = 6, data_folder_path = "data/DTI_23_series")
    
    n_ds = length(readdir(data_folder_path))

    df = Vector{DataFrame}(undef, n_ds)
    t = Vector{Vector{Float64}}(undef, n_ds)
    time = Vector{Vector{DateTime}}(undef, n_ds)
    Y = Vector{Matrix}(undef, n_ds)
    U = Vector{Matrix}(undef, n_ds)
    U_raw = Vector{Matrix}(undef, n_ds)
    D = Vector{Matrix}(undef, n_ds)
    D_raw = Vector{Matrix}(undef, n_ds)
    misc = Vector{Matrix}(undef, n_ds)
    camp = Vector{Int}(undef, n_ds)
    series = Vector{Int}(undef, n_ds)

    for i in 1:n_ds
        df[i] = CSV.read("$(data_folder_path)/series_$(i).csv", DataFrame)
        camp[i] = find_camp(df[i].time[1])
        time[i] = df[i].time[1:every_k:end]
        t[i] = df[i].t[1:every_k:end] .- df[i].t[1] # Shift time to start at 0
        Y[i] = hcat(df[i].ygA, df[i].TlA_HX_out, df[i].TlD_HX_out, df[i].P_reb)[1:every_k:end,:]'
       
        # Smooth variables using moving average
        cgina_smoothed = moving_average(df[i].cginA, 30)
        Fco_smoothed = moving_average(df[i].Fco, 10)

        U_raw[i] =  hcat(Fco_smoothed, df[i].TlA_HX_out_SP, df[i].TlD_HX_out_SP)[1:every_k:end,:]' # Raw inputs (before smoothing)

        Fco_sp = find_input_sp(Fco_smoothed, 0.1)
        U[i] = hcat(Fco_sp, df[i].TlA_HX_out_SP, df[i].TlD_HX_out_SP)[1:every_k:end,:]' # Smoothed Fco and SP for TlD_HX
        #U[i] = hcat(df[i].Fco, df[i].TlA_HX_out_SP, df[i].TlD_HX_out_SP)[1:every_k:end,:]' # Raw input data

        D_raw[i] = hcat(cgina_smoothed, df[i].FginA)[1:every_k:end,:]'

        cgina_sp = find_input_sp(cgina_smoothed, 0.10)
        Fgina_sp = find_input_sp(df[i].FginA, 10.0)
        D[i] = hcat(cgina_sp, Fgina_sp)[1:every_k:end,:]' # Averaged disturbances
        #D[i] = hcat(cgina_smoothed, df[i].FginA)[1:every_k:end,:]' # Smoothed cgina, but not Fgina, as it is already quite smooth

        # Na: Capture rate
        cga = df[i].cgA
        Na = Fgina_sp .* (cgina_sp - cga) # Mass balance, CO2, gas phase, absorber, steady state
        misc[i] = hcat(Na)[1:every_k:end]'
    end

    # Loop through the 4 campaigns and assign pr. campaign series number to each dataset
    camp_series = [0, 0, 0, 0] # Counters for each campaign
    for i in 1:n_ds
        camp_series[camp[i]] += 1
        series[i] = camp_series[camp[i]]
    end

    datasets = [ (t = t[i], time = time[i], Y = Y[i], u = U[i], u_raw = U_raw[i], dist = D[i], dist_raw = D_raw[i], N = length(t[i]), misc = misc[i], camp = camp[i], series = series[i]) for i in 1:n_ds ]
    return datasets, df
end

function find_input_sp(y, threshold)
    z = fill(0.0, length(y))
    
    sp_idx_start = 1
    current_z = y[sp_idx_start]
    for i in eachindex(y[2:end])
        if abs(y[i] - current_z) > threshold # Detect change point
            
            z[sp_idx_start:i-1] .= current_z # Assign the previous mean value to the segment
            sp_idx_start = i # Update the start index for the new segment
        end

        current_z = mean(y[sp_idx_start:i]) # Update the current mean value
    end
    z[sp_idx_start:end] .= current_z # Assign the last segment

    return z
end

function find_camp(time)
    if month(time) in [5] && year(time) == 2024
        return 1
    elseif  month(time) in [6] && year(time) == 2024
        return 2
    elseif month(time) in [3] && year(time) == 2025
        return 3
    elseif month(time) in [11,12] && year(time) == 2025
        return 4
    else
        return NaN
    end
end



# Model:

# function ffun_bar(x,u,d,p,p_fixed)
#     # Extract states
#     ca, cd, ct = x
    
#     Va = 0.17
#     Vd = 0.17
#     Vt = 0.17

#     # Conversion Constants
#     M_CO2 = 44.01 # g/mol
#     g2kg = 1e-3 # g to kg
#     mol2kmol = 1e-3

#     # Inputs #
#     F, Q = u

#     # Disturbances #
#     #cgina, Fgina = d(t)

#     # Evaluate Na
#     input_Na = [ca]
#     Na = evaluate_NN(p.Na_params, p_fixed.Na, input_Na)[1] # Output is scalar

#     # Evaluate Nd
#     input_Nd = [ca, cd, F, Q]
#     Nd = evaluate_NN(p.Nd_params, p_fixed.Nd, input_Nd)[1] # Output is scalar
#     # ODEs:
#     dx = [(F * (ct - ca) + Na * mol2kmol) / Va,
#         (F * (ca - cd) - Nd * mol2kmol) / Vd,
#         F * (cd - ct) / Vt
#     ] 
    
#     return dx
# end

# function ffun_bar_DTI(x,u,d,p,p_fixed)
#     # Extract states
#     ca, cd, TlD_HX = x
    
#     Va = 0.25
#     Vd = 0.25

#     # Conversion Constants
#     M_CO2 = 44.01 # g/mol
#     g2kg = 1e-3 # g to kg
#     mol2kmol = 1e-3

#     # Inputs #
#     F, TlD_HX_SP = u

#     # Disturbances #
#     #cgina, Fgina = d(t)

#     # Evaluate Na
#     input_Na = [ca]
#     Na = evaluate_NN(p.Na_params, p_fixed.Na, input_Na)[1] # Output is scalar

#     # Evaluate Nd
#     input_Nd = [ca, cd, F, TlD_HX]
#     Nd = evaluate_NN(p.Nd_params, p_fixed.Nd, input_Nd)[1] # Output is scalar
#     # ODEs:
#     dx = [(F * (cd - ca) + Na * mol2kmol) / Va,
#         (F * (ca - cd) - Nd * mol2kmol) / Vd,
#         p.theta * (TlD_HX_SP - TlD_HX) # Simple first-order dynamics for TlD_HX
#     ] 
    
#     return dx
# end

# function ffun_bar_DTI_DAE(x, u, d, p, p_fixed)

#     ca, cd, TlD_HX, cga = x

#     Va = 0.25
#     Vd = 0.25

#     # Conversion Constants
#     M_CO2 = 44.01 # g/mol
#     g2kg = 1e-3 # g to kg
#     mol2kmol = 1e-3

#     # Inputs #
#     F, TlD_HX_SP = u

#     # Disturbances #
#     cgina, Fgina = d

#     # Evaluate Na
#     input_Na = [ca, cga]
#     Na = evaluate_NN(p.Na_params, p_fixed.Na, input_Na)[1] # Output is scalar

#     # Evaluate Nd
#     input_Nd = [ca, cd, F, TlD_HX]
#     Nd = evaluate_NN(p.Nd_params, p_fixed.Nd, input_Nd)[1] # Output is scalar
#     # ODEs:
#     dx = [(F * (cd - ca) + Na * mol2kmol) / Va,
#         (F * (ca - cd) - Nd * mol2kmol) / Vd,
#         p.theta * (TlD_HX_SP - TlD_HX), # Simple first-order dynamics for TlD_HX
#         Fgina * (cgina - cga) - Na # Algebraic equation for cga, mass balance, absorber gas phase, steady state
#     ] 
    
#     return dx

# end
# function ffun_DTI(x, u, d, p, t, p_fixed)
#     return ffun_bar_DTI(x, u(t), d(t), p, p_fixed)
# end

# function ffun_DTI_DAE(x, u, d, p, t, p_fixed)
#     return ffun_bar_DTI_DAE(x, u(t), d(t), p, p_fixed)
# end


# function ffun(x, u, d, p, t, p_fixed)
    
#     return ffun_bar(x, u(t), d(t), p, p_fixed)                   

# end

function ffun_Jac(x, u, d, p, t, p_fixed, ffun)

    return ForwardDiff.jacobian(_x -> ffun(_x, u, d, p, t, p_fixed), x)

end



# Create ODE problem for 1. and 2. moment equations for each dataset
function moment_equations(x, u, dist, p, t, p_fixed, ffun)

    # Number of states
    nx = p_fixed.nx
    # Extract mean and covariance from x
    x_mean = x[1:nx]
    P = reshape(x[nx+1:end], (nx, nx))
    # Extract noise parameters from p
    log_sx = p.log_sx
    sx = exp.(log_sx)

    f = ffun(x_mean, u, dist, p, t, p_fixed)

    # Linearize f around current state to get A matrix (df/dx)
    A = ForwardDiff.jacobian(_x -> ffun(_x, u, dist, p, t, p_fixed), x_mean)
    #A = ffun_Jac(x_mean, u, dist, p, t, p_fixed)
    dPdt = A * P + P * A' + sx.^2 .* I(nx) # Process noise added to covariance dynamics

    return vcat(f, reshape(dPdt, :))

end

# function hfun(x, u, d, p, p_fixed)
#     # Extract states
#     ca = x[1]

#     # Conversion Constants
#     R = 8.314 # J/(mol*K)
#     Ta = 313.15 # K
#     Pa = 101325 # Pa

#     # Inputs #
#     F, Q = u

#     # Disturbances #
#     cgina, Fgina = d

#     # Evaluate Na
#     input_Na = [ca]
#     Na = evaluate_NN(p.Na_params, p_fixed.Na, input_Na)[1] # Output is scalar

#     # Mass balance, CO2, gas phase, absorber, steady state:
#     cga = cgina - Na / Fgina
    
#     # Total mol concentration in gas phase (assuming ideal gas behavior and constant T and P)
#     cga_tot = Pa / (R * Ta)

#     # Output CO2 concentration [vol%]:
#     yga = cga / cga_tot * 100

#     return [yga]
# end

# function hfun_DTI(x, u, d, p, p_fixed)
#     # Extract states
#     ca, cd, TlD_HX = x

#     # Conversion Constants
#     R = 8.314 # J/(mol*K)
#     Ta = 313.15 # K
#     Pa = 101325 # Pa

#     # Inputs #
#     F, TlD_HX_SP = u

#     # Disturbances #
#     cgina, Fgina = d

#     # Evaluate Na
#     input_Na = [ca]
#     Na = evaluate_NN(p.Na_params, p_fixed.Na, input_Na)[1] # Output is scalar

#     # Evaluate Nd 
#     input_Nd = [ca, cd, F, TlD_HX]
#     Nd = evaluate_NN(p.Nd_params, p_fixed.Nd, input_Nd)[1] # Output is scalar

#     # Mass balance, CO2, gas phase, absorber, steady state:
#     cga = cgina - Na / Fgina
    
#     # Total mol concentration in gas phase (assuming ideal gas behavior and constant T and P)
#     cga_tot = Pa / (R * Ta)

#     # Output CO2 concentration [vol%]:
#     yga = cga / cga_tot * 100

#     # Reboiler duty [kW]
#     input_P_reb = [ca, cd, F, TlD_HX, TlD_HX - TlD_HX_SP] # Maybe consider using Nd as input to this NN
#     P_reb = evaluate_NN(p.P_reb_params, p_fixed.P_reb, input_P_reb)[1] # Output is scalar
#     return [yga, TlD_HX, P_reb, Na, Nd]
# end

# function hfun_obs_DTI(x, u, d, p, p_fixed)
#     return hfun_DTI(x, u, d, p, p_fixed)[1:3]
# end

# function hfun_obs_DTI_DAE(x, u, d, p, p_fixed)
#     return hfun_DTI_DAE(x, u, d, p, p_fixed)[1:3]
# end


# function hfun_DTI_DAE(x, u, d, p, p_fixed)
#     # Extract states
#     ca, cd, TlD_HX, cga = x

#     # Conversion Constants
#     R = 8.314 # J/(mol*K)
#     Ta = 313.15 # K
#     Pa = 101325 # Pa

#     # Inputs #
#     F, TlD_HX_SP = u

#     # Disturbances #
#     cgina, Fgina = d

#     # Evaluate Na
#     input_Na = [ca, cga]
#     Na = evaluate_NN(p.Na_params, p_fixed.Na, input_Na)[1] # Output is scalar

#     # Evaluate Nd 
#     input_Nd = [ca, cd, F, TlD_HX]
#     Nd = evaluate_NN(p.Nd_params, p_fixed.Nd, input_Nd)[1] # Output is scalar

#     # Mass balance, CO2, gas phase, absorber, steady state:
#     #cga = cgina - Na / Fgina
    
#     # Total mol concentration in gas phase (assuming ideal gas behavior and constant T and P)
#     cga_tot = Pa / (R * Ta)

#     # Output CO2 concentration [vol%]:
#     yga = cga / cga_tot * 100

#     # Reboiler duty [kW]
#     input_P_reb = [ca, cd, F, TlD_HX, TlD_HX - TlD_HX_SP] # Maybe consider using Nd as input to this NN
#     P_reb = evaluate_NN(p.P_reb_params, p_fixed.P_reb, input_P_reb)[1] # Output is scalar
#     return [yga, TlD_HX, P_reb, Na, Nd]
# end

# function hfun_detailed(x, u, d, p, p_fixed)
#     # Extract states
#     ca, cd, ct = x
    
#    # Conversion Constants
#     R = 8.314 # J/(mol*K)
#     Ta = 313.15 # K
#     Pa = 101325 # Pa
#     M_CO2 = 44.01 # g/mol
#     g2kg = 1e-3 # g to kg
#     mol2kmol = 1e-3

#     # Inputs #
#     F, Q = u

#     # Disturbances #
#     cgina, Fgina = d

#     # Evaluate Na
#     input_Na = [ca]
#     Na = evaluate_NN(p.Na_params, p_fixed.Na, input_Na)[1] # Output is scalar
    
#     # Evaluate Nd
#     input_Nd = [ca, cd, F, Q]
#     Nd = evaluate_NN(p.Nd_params, p_fixed.Nd, input_Nd)[1] # Output is scalar

#     # Mass balance, CO2, gas phase, absorber, steady state:
#     cga = cgina - Na / Fgina
    
#     # Total mol concentration in gas phase (assuming ideal gas behavior and constant T and P)
#     cga_tot = Pa / (R * Ta)

#     # Inlet CO2 concentration [vol%]:
#     ygina = cgina / cga_tot * 100

#     # Output CO2 concentration [vol%]:
#     yga = cga / cga_tot * 100

#     # Capture efficiency:
#     cap_eff = (ygina - yga) / ygina * 100

#     # Capture rate [kg/h]
#     ma = Na * M_CO2 * g2kg

#     z = [yga, ygina, cap_eff, ma, Na, Nd]

#     return z
# end

function hfun_Jac(x, u, d, p, p_fixed)
    
    return ForwardDiff.jacobian(_x -> hfun(_x, u, d, p, p_fixed), x)

end

# Evaluate NN
function evaluate_NN(NN_params, NN_normalization, z0)
    mu_input        = NN_normalization.input.mu
    sigma_input     = NN_normalization.input.sigma
    mu_output       = NN_normalization.output.mu
    sigma_output    = NN_normalization.output.sigma

    z0_normalized   = (z0 .- mu_input) ./ sigma_input

    n_layers        = length(keys(NN_params))
    layer = 0
    z = z0_normalized

    for i in 1:(n_layers-1) # Exclude input and output pars and output layer
        layer += 1
        W = getproperty(NN_params, Symbol("layer_", layer)).W
        b = getproperty(NN_params, Symbol("layer_", layer)).b
        z = tanh.(W * z + b)
    end

    # Output layer
    W = getproperty(NN_params, Symbol("layer_", layer + 1)).W
    b = getproperty(NN_params, Symbol("layer_", layer + 1)).b
    z = W * z + b

    # Re-scale output
    z_output = z .* sigma_output .+ mu_output

    return z_output
end

function evaluate_NN_outer(NN_params, NN_settings, x, u, d)

    input_func = NN_settings.input_func
    input_idxs = NN_settings.input_idxs

    input_raw = vcat(x[input_idxs.x], u[input_idxs.u], d[input_idxs.d])
    input = input_func(input_raw)
    return evaluate_NN(NN_params, NN_settings, input)[1] # Output is scalar
end




# function get_NN_settings(; model = "tiller")
    

#     if model == "tiller"
#         # NN settings:
#         # Na NN:
#         # Specify number of neurons in each layer:
#         n_neurons_Na = Float64[] # (No hidden layers)
#         mu_input_Na = [2.60]
#         sigma_input_Na = [0.05]
#         mu_output_Na = [648]
#         sigma_output_Na = [48]

#         Na = (input = (mu = mu_input_Na, sigma = sigma_input_Na),
#             output = (mu = mu_output_Na, sigma = sigma_output_Na), n_neurons = n_neurons_Na)

#         input_names_Nd = ["ca", "cd", "F", "Q"]
#         output_names_Nd = ["Nd"]
#         # Specify number of neurons in each layer:
#         n_neurons_Nd = [8] # 1 hidden layer with 8 neurons

#         # Normalization parameters for input and output, Nd
#         mu_input_Nd = [2.60, 0.3, 0.3, 28]
#         sigma_input_Nd = [0.05, 0.22, 0.10, 3.95]
#         mu_output_Nd = [648]
#         sigma_output_Nd = [48]

#         Nd = (input = (mu = mu_input_Nd, sigma = sigma_input_Nd),
#             output = (mu = mu_output_Nd, sigma = sigma_output_Nd), n_neurons = n_neurons_Nd)

#         NN_settings = (Na = Na, Nd = Nd)


#     elseif model == "DTI"
#         # NN settings:


#         # 26-02-26 09-33: Changed sigma_ca and sigma_cd from 0.05 to 0.1.
#                                # and sigma_Na and sigma Nd from 100 to 50.

#         # Na NN:
#         # Inputs: ca, Fgina, cgina
#         # Specify number of neurons in each layer:
#         n_neurons_Na = Int[4] # (No hidden layers)
#         # Normalization parameters for input and output
#         mu_input_Na = [2.60, 100, 6.0]
#         sigma_input_Na = [0.1, 30.0, 0.5]
#         mu_output_Na = [300]
#         sigma_output_Na = [50]

#         Na = (input = (mu = mu_input_Na, sigma = sigma_input_Na),
#                 output = (mu = mu_output_Na, sigma = sigma_output_Na), n_neurons = n_neurons_Na)

#         #Nd :
#         input_names_Nd = ["ca", "cd", "F", "TlD_HX_out"]
#         output_names_Nd = ["Nd"]
#         # Specify number of neurons in each layer:
#         n_neurons_Nd = [4] 

#         # Normalization parameters for input and output, Nd
#         mu_input_Nd = [2.60, 2.10, 0.5, 90]
#         sigma_input_Nd = [0.1, 0.1, 0.2, 10]
#         mu_output_Nd = [300]
#         sigma_output_Nd = [50] 

#         Nd = (input = (mu = mu_input_Nd, sigma = sigma_input_Nd),
#             output = (mu = mu_output_Nd, sigma = sigma_output_Nd), n_neurons = n_neurons_Nd)

      

#         # P_reb NN:
#         n_neurons_P_reb = [4] 

#         # Normalization parameters for input and output, P_reb
#         mu_input_P_reb = [2.60, 2.10, 0.5, 90, 0.0]
#         sigma_input_P_reb = [0.1, 0.1, 0.2, 10.0, 2.0]
#         mu_output_P_reb = [15]
#         sigma_output_P_reb = [5]

#         P_reb = (input = (mu = mu_input_P_reb, sigma = sigma_input_P_reb),
#             output = (mu = mu_output_P_reb, sigma = sigma_output_P_reb), n_neurons = n_neurons_P_reb)

#         NN_settings = (Na = Na, Nd = Nd, P_reb = P_reb)

#     else
#         error("Model $(model) not recognized. Valid options are: 'tiller', 'DTI'.")
#     end

    

 
#     return NN_settings
# end

function initialize_NN_params_from_settings(nn_setting)
    n_neurons = nn_setting.n_neurons
    n_input   = length(nn_setting.input.mu)
    n_output  = length(nn_setting.output.mu)

    layer_sizes = [n_input; Int.(n_neurons); n_output]
    n_layers    = length(layer_sizes) - 1

    layer_params = []
    for i in 1:n_layers
        W = randn(layer_sizes[i+1], layer_sizes[i]) .* 0.1
        b = zeros(layer_sizes[i+1])
        push!(layer_params, (W = W, b = b))
    end

    layer_nt = NamedTuple{Tuple(Symbol("layer_", i) for i in 1:n_layers)}(
        Tuple(layer_params)
    )

    return layer_nt  # pure NamedTuple — safe for ComponentArray
end

function get_NN_normalization(NN_settings)
    return (
        Na = (input  = NN_settings.Na.input,  output = NN_settings.Na.output),
        Nd = (input  = NN_settings.Nd.input,  output = NN_settings.Nd.output)
    )
end


## Prediction step for dataset 1 from time 1 to time 2
# function ekf_predict(prob, x0, p, t_span, sensealg, p_fixed)

#     nx = p_fixed.nx

#     # _prob = remake(prob, u0=x0, p=p, tspan=t_span)
#     # sol = Array(solve(_prob, Tsit5();
#     #     tstops  = t_span,
#     #     saveat  = t_span,
#     #     abstol  = 1e-6,
#     #     reltol  = 1e-6,
#     #     sensealg = sensealg))

#     # x_prior = sol[1:nx, end]
#     # P_prior = reshape(sol[nx+1:end, end], (nx, nx))
#     _prob = remake(prob, u0=x0, p=p, tspan=t_span)
#     sol = solve(_prob, Tsit5();
#         tstops  = [t_span[2]],
#         saveat  = [t_span[2]],
#         abstol  = 1e-6,
#         reltol  = 1e-6,
#         sensealg = sensealg)

#     x_prior = sol[1:nx, end]
#     P_prior = reshape(sol[nx+1:end, end], (nx, nx))    

#     return x_prior, P_prior
# end

# function ekf_update(hfun_ekf, x_prior, P_prior, y_meas, u, dist, p, p_fixed)

#     nx = p_fixed.nx
#     ny = p_fixed.ny
#     sy = p_fixed.sy

#     hfun = p_fixed.hfun

#     h = hfun_ekf(x_prior, u, dist, p, p_fixed) # Predicted measurement
#     e = y_meas - h # Innovation error

#     # dhdx
#     H = ForwardDiff.jacobian(x -> hfun_ekf(x, u, dist, p, p_fixed), x_prior)
#     #H = hfun_Jac(x_prior, u, dist, p, p_fixed)

#     Rv = Symmetric(sy.^2 .* I(ny)) # Measurement noise covariance

#     R = Symmetric(H * P_prior * H' + Rv) # Innovation covariance
#     R_chol = cholesky(R)
#     K = (R_chol \ (H * P_prior))' # Kalman gain using Cholesky factorization for numerical stability

#     IKH = I(nx) - K * H
#     x_post = x_prior + K * e # Updated state estimate
#     P_post = Symmetric(IKH * P_prior * IKH' + K * Rv * K') # Updated covariance estimate using Joseph form for numerical stability

#     return x_post, P_post, R_chol, e

# end

function loss_NLL(p, x0, ds, prob, sensealg, p_fixed)

    NLL = zero(eltype(p))

    x = copy(x0)
    P = copy(p_fixed.P0)

    nx = p_fixed.nx
    hfun_ekf = p_fixed.hfun
    obs_idx = p_fixed.obs_idx

    x0_P0 = similar(x0, nx + nx^2) # Allocate memory for combined state and covariance vector

    # Step through each time step in the dataset
    for k in 1:(length(ds.t)-1)

            # Prediction step
            x0_P0[1:nx] .= x
            x0_P0[nx+1:end] .= reshape(P, nx^2)

            x_prior, P_prior = ekf_predict(prob, x0_P0, p, (ds.t[k], ds.t[k+1]), sensealg, p_fixed)
    
            # Update step with measurement at time k+1
            y_meas = ds.Y[obs_idx,k+1]
            u = ds.u[:,k+1]
            dist = ds.dist[:,k+1]
    
            x_post, P_post, R_chol, e = ekf_update(hfun_ekf, x_prior, P_prior, y_meas, u, dist, p, p_fixed)
    
            # Compute NLL contribution from this time step
            NLL += 0.5 * (2 * sum(log.(diag(R_chol.L))) + sum(abs2, R_chol.L \ e))
    
            # Update state and covariance for next iteration
            x = x_post
            #P = P_post
            P = Symmetric((P_post + P_post') / 2) + 1e-4 * I(nx) # Ensure P stays symmetric and positive definite
        end
    return NLL
end

function NLL_aggregate(θ, datasets, prob_list, sensealg, p_fixed)
    p = θ.p
    x0_vec = θ.x0
    p_NN = θ[p_fixed.idx_NN]

    NLL_parts = zeros(eltype(p), length(datasets))

    Threads.@threads for i in eachindex(datasets)
        ds = datasets[i]
        NLL_parts[i] = loss_NLL(p, x0_vec[i], ds, prob_list[i], sensealg, p_fixed)
    end

    NLL_aggregate_value = sum(NLL_parts) / length(datasets)

    par_reg = p_fixed.lambda_reg * sum(p_NN .^ 2)

    loss = NLL_aggregate_value + par_reg

    return loss
end


callback_opt_02 = (θ, loss, stop_crit, early_stop_window) -> begin
    push!(loss_history, loss)
    
    # Print every 10 iterations
    if length(loss_history) % 5 == 0
        println("Iter $(length(loss_history)): loss = $loss")
    end
    flush(stdout)
    # Early stopping: stop if loss hasn't improved in 10 iters
    if length(loss_history) > early_stop_window
        recent_improvement = loss_history[end-20] - loss
        if recent_improvement < stop_crit
            println("Early stopping at iter $(length(loss_history))")
            println("Recent improvement: ", recent_improvement)
            # Print the final 25 loss values
            for (i, l) in enumerate(loss_history[end-25:end])
                println("Iter $(length(loss_history)-25+i): loss = $(l)")
            end
            flush(stdout)
            return true  # stop optimizer
        end
    end
    
    false
end


function euler_simulation(ffun, hfun, x0, U, D, p, p_fixed, t_vec, dt_sim)
    nx = length(x0)
    X = zeros(nx, length(t_vec))
    X[:,1] = x0
    
    z0 = hfun(x0, U[:,1], D[:,1], p, p_fixed)
    nz = length(z0)
    Z = zeros(nz, length(t_vec)-1)
    
    for i in 1:(length(t_vec)-1)
        f = ffun(X[:,i], U[:,i], D[:,i], p, p_fixed)
        X[:,i+1] = X[:,i] + dt_sim * f
        Z[:,i] = hfun(X[:,i], U[:,i], D[:,i], p, p_fixed)
    end
    return X, Z
end


function print_optimizer_stats(res)
    stats = res.stats
    println("========== Optimization Summary ==========")
    println("Iterations:           ", stats.iterations)
    println("Time (s):             ", @sprintf("%.1f", stats.time))
    println("Function evals:       ", stats.fevals)
    println("Gradient evals:       ", stats.gevals)
    println("Hessian evals:        ", stats.hevals)
    println("Final loss:           ", @sprintf("%.3e", res.objective))
    println("retcode: ", res.retcode)
    println("========================================")
end

function get_NN_idxs(NN_settings, θ)
    ax_p = getaxes(θ.p)[1]
    ax_theta = getaxes(θ)[1]
    p_range = ax_theta[:p].idx
    p_start = first(p_range)

    idx_NN = Int[]

    for k in keys(NN_settings)
        nn = NN_settings[k]

        # Only include if it's NOT a linear transformation
        if !isempty(nn.n_neurons)
            idxs = ax_p[Symbol(string(k, "_params"))].idx .+ p_start .- 1
            append!(idx_NN, idxs)
        end
    end

    return idx_NN
end

function get_NN_idxs_W_only(NN_settings, θ)
    ax_p = getaxes(θ.p)[1]
    ax_theta = getaxes(θ)[1]
    p_range = ax_theta[:p].idx
    p_start = first(p_range)

    idx_NN = Int[]

    for k in keys(NN_settings)
        nn = NN_settings[k]

        # Only include if it's NOT a linear transformation
        if !isempty(nn.n_neurons)
            idxs = ax_p[Symbol(string(k, "_params"))].idx .+ p_start .- 1
            append!(idx_NN, idxs)
        end
    end

    return idx_NN
end

function moving_average(x::AbstractVector, window::Int)
    @assert window ≥ 1 "Window size must be ≥ 1"
    @assert window ≤ length(x) "Window size must be ≤ length of data"

    n = length(x)
    half = div(window, 2)

    y = similar(x, Float64)

    for i in 1:n
        left  = max(1, i - half)
        right = min(n, i + half)
        y[i] = mean(@view x[left:right])
    end

    return y
end


function optimize_2_step(loss_fn, θ0)

    # Optimizer settings
    ad_type = Optimization.AutoForwardDiff()

    # Optimization function
    optf = Optimization.OptimizationFunction(loss_fn, ad_type)

    # Step 1: Adam optimizer
    optimizer_adam = OptimizationOptimisers.ADAM(0.05)
    prob_adam = Optimization.OptimizationProblem(optf, θ0)

    # Callback Adam optimizer
    function cb_adam(θ, loss)
        tol = 1e-8
        return callback_opt_02(θ, loss, tol, 50)
    end

    println("\nStep 1: Optimizing with ADAM...")
    result_adam = Optimization.solve(prob_adam, optimizer_adam;
        #maxiters = 1500,
        #maxiters = 1000,
        maxiters = 2500,
        #maxiters = 10,
        #abstol = 1e-6,
        #reltol = 1e-6,
        callback = cb_adam)

    adam_iters = length(loss_history)

    # Callback lbfgs optimizer
    function cb_lbfgs(θ, loss)
        tol = 1e-8
        #tol = 1e-4
        return callback_opt_02(θ, loss, tol, 50 + adam_iters) # continue counting iterations from Adam
    end

    #Step 2: LBFGS optimizer
    optimizer_lbfgs = OptimizationOptimJL.LBFGS(
        linesearch = LineSearches.BackTracking()
    )
    prob_lbfgs = Optimization.OptimizationProblem(optf, result_adam.u)

    println("\nStep 2: LBFGS...")
    lbfgs_succeeded = false
    result_lbfgs = nothing
    try
        result_lbfgs = Optimization.solve(prob_lbfgs, optimizer_lbfgs;
            #maxiters = 500,
            #maxiters = 2500,
            maxiters = 5000,
            abstol = 1e-8,
            reltol = 1e-8,
            #maxiters = 1500, 
            #maxiters = 10,
            callback = cb_lbfgs)
        lbfgs_succeeded = true
        println("LBFGS optimization completed successfully.")
    catch e
        @warn "LBFGS optimization failed with error: $e. Proceeding with ADAM results."
    end

    # Use LBFGS results if successful, otherwise fall back to ADAM results
    if lbfgs_succeeded
        final_sol = result_lbfgs.u # change to result_lbfgs.u if running LBFGS
        stats = (adam = result_adam.stats, lbfgs = result_lbfgs.stats)
        println("\nOptimization completed (Adam + LBFGS).")
        println("\nADAM:"); print_optimizer_stats(result_adam)
        println("\nLBFGS:"); print_optimizer_stats(result_lbfgs)
    else
        final_sol = result_adam.u # fall back to ADAM results if LBFGS failed
        stats = (adam = result_adam.stats, lbfgs = nothing)
        println("\nOptimization completed (Adam only — LBFGS failed).")
        println("\nADAM:"); print_optimizer_stats(result_adam)
    end

    p_opt = NamedTuple(final_sol.p)
    x0_opt = copy.(final_sol.x0)

    adam_sol = result_adam.u
    p_opt_adam = NamedTuple(adam_sol.p)
    x0_opt_adam = copy.(adam_sol.x0)

    return p_opt, x0_opt, stats, loss_history, p_opt_adam, x0_opt_adam
end


function get_input_and_disturbance_bounds(datasets)
    U_min = minimum(hcat([minimum(dataset.u, dims = 2) for dataset in datasets]...), dims = 2)
    U_max = maximum(hcat([maximum(dataset.u, dims = 2) for dataset in datasets]...), dims = 2)

    D_min = minimum(hcat([minimum(dataset.dist, dims = 2) for dataset in datasets]...), dims = 2)
    D_max = maximum(hcat([maximum(dataset.dist, dims = 2) for dataset in datasets]...), dims = 2)

    return U_min, U_max, D_min, D_max
end


function get_ss_data(datasets, output_var, plant, w_Ta, norm_params)
    inputs_ss = []
    output_ss = []
    k = 5
    for i in eachindex(datasets)
        ds = datasets[i]
        output_var_i = output_var[i]

        if plant == "DTI"
            ca_ss = norm_params[:ca].mu
            cd_ss = norm_params[:cd].mu
            Tla_HX_ss = mean(ds.Y[2, 1:k])
            Tld_HX_ss = mean(ds.Y[3, 1:k])
            if w_Ta
                x_ss = [ca_ss, cd_ss, Tla_HX_ss, Tld_HX_ss]
            else
                x_ss = [ca_ss, cd_ss, Tld_HX_ss]
            end
        elseif plant == "Tiller"
            ca_ss = norm_params[:ca].mu
            cd_ss = norm_params[:cd].mu
            ct_ss = norm_params[:ct].mu
            x_ss = [ca_ss, cd_ss, ct_ss]
        end

        u_ss_0 = mean(ds.u[:, 1:k], dims=2)
        d_ss_0 = mean(ds.dist[:, 1:k], dims=2)

        u_ss_f = mean(ds.u[:, end-k+1:end], dims=2)
        d_ss_f = mean(ds.dist[:, end-k+1:end], dims=2)
        
        z_ss_0 = mean(output_var_i[1:k])
        z_ss_f = mean(output_var_i[end-k+1:end])

        push!(inputs_ss, (x = x_ss, u = u_ss_0, d = d_ss_0))
        push!(inputs_ss, (x = x_ss, u = u_ss_f, d = d_ss_f))
        push!(output_ss, z_ss_0)
        push!(output_ss, z_ss_f)
    end

    return inputs_ss, output_ss

end




# Evaluate NN
function loss_ss(p, input_ss, output_ss, NN_fixed)
    NN_par_ss = p.NN_par_ss
    loss_NN = zeros(eltype(p), length(input_ss))
    for i in eachindex(input_ss)
        
        NN_ss_out_i = evaluate_NN_outer(NN_par_ss, NN_fixed, input_ss[i].x, input_ss[i].u, input_ss[i].d)[1]
        
        loss_NN[i] = (NN_ss_out_i .- output_ss[i]).^2
        #println("Dataset $(i), NN_ss: ", NN_ss_i, "output_ss: ", output_ss[i])
    end
    return mean(loss_NN)
end

# Evaluate Na loss on steady-state
function loss_Na_ss(p, input_ss, output_ss, p_fixed)
    
    eval_Na_fun = p_fixed.eval_Na_fun
    Na_params = p.NN_par_ss
    loss_Na = zeros(eltype(p), length(input_ss))
    
    for i in eachindex(input_ss)
        x = input_ss[i].x
        u = input_ss[i].u
        d = input_ss[i].d
    
        Na = eval_Na_fun(Na_params, p_fixed, x, u, d)[1]

        loss_Na[i] = (Na .- output_ss[i]) .^ 2
    end
    return mean(loss_Na)
end




function compute_init_par_ss(datasets, output_var, NN_par_ss_init, NN_fixed, plant, w_Ta, loss_func, norm_params)
    
    input_ss, output_ss = get_ss_data(datasets, output_var, plant, w_Ta, norm_params);

    # Initial parameters for optimization
    p_CA = ComponentArray(NN_par_ss = NN_par_ss_init)

    loss_ss_fn = (θ, _) -> loss_func(θ, input_ss, output_ss, NN_fixed)

    # Optimizer settings
    ad_type = Optimization.AutoForwardDiff()

    # Optimization function
    optf = Optimization.OptimizationFunction(loss_ss_fn, ad_type)

    # Step 1: Adam optimizer
    optimizer_adam = OptimizationOptimisers.ADAM(0.05)
    prob_adam = Optimization.OptimizationProblem(optf, p_CA)

    # Callback Adam optimizer
    function cb_adam(θ, loss)
        return callback_opt_02(θ, loss, 1e-4, 50)
    end

    global loss_history = Float64[]

    #println("\nStep 1: Optimizing with ADAM...")
    result_adam = Optimization.solve(prob_adam, optimizer_adam;
        #callback = cb_adam,
        maxiters = 1000)

    #println("NN pars initialised by ADAM optimization on steady-state data:")
    #println("Obj: ", result_adam.objective)
   # println("Stats: ", result_adam.stats)

    NN_par_ss = result_adam.u.NN_par_ss

    return NN_par_ss
end

function gfun(z, F, d, p, p_fixed, cap_eff_target, ffun, hfun)

    u_bar = [F; NaN; z[4]] # NaN for TlA_SP, which is not used in the model
    f = ffun(z[1:3], u_bar, d, p, p_fixed)
    h = hfun(z[1:3], u_bar, d, p, p_fixed)

    h_named = named_output(h)

    cap_eff = h_named.cap_eff[1] # Capture efficiency is the 6th output of hfun

    g = vcat(f, cap_eff - cap_eff_target) # Combine f and the capture efficiency constraint

    return g
end

function ffun_with_accum_CO2(x, u, d, p, p_fixed, ffun, hfun)
   
    M_CO2 = 44.01 # g/mol
    g2kg = 1e-3 # g to kg

    cgina, Fgina = d

    f = ffun(x[1:end-2], u, d, p, p_fixed)
    h = hfun(x[1:end-2], u, d, p, p_fixed)

    Na = h[5]

    ma = Na * M_CO2 * g2kg                  # CO2 capture rate [kg/h]
    mina = cgina * Fgina * M_CO2 * g2kg     # CO2 inlet mass flow rate [kg/h]
   
    return vcat(f, ma, mina)

end

function ffun_plant_model_mismatch(x, u, d, p, p_fixed, ffun_model)
    # Get model prediction for state derivative
    f_model = ffun_model(x, u, d, p, p_fixed)

    # Introduce a model mismatch
    #f_model[2] += 0.5 + 0.1 * rand()
    f_model[1] += 0.25
    return f_model
end

function get_steady_state(ffun, x_init, u_ss, dist_ss, p, p_fixed)
    t_vec_ss = 0:20/3600:3
    U_ss = repeat(u_ss, 1, length(t_vec_ss))
    D_ss = repeat(dist_ss, 1, length(t_vec_ss))
    X_ss  = simulate_system(ffun, x_init, U_ss, D_ss, p, p_fixed, t_vec_ss)
    sol   = nlsolve(x -> ffun(x, u_ss, dist_ss, p, p_fixed), X_ss[:,end])
    x_ss  = sol.zero
    return x_ss
end