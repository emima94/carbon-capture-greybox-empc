
# One unified function to get the model. 
function get_model(datasets, NN_def, plant, w_Ta, with_Q, eval_Na_fun, norm_params_func; lambda_reg = 1e-4, obs_weights = nothing, ode_solver_str = "Tsit5", compute_ss_params = true)

    norm_params = norm_params_func(plant)
    # Get NN settings
    NN_settings = get_NN_settings(NN_def, plant, w_Ta, norm_params)

    # Observation idxs from output function hfun
    obs_idx = get_obs_idx(plant, w_Ta, with_Q)

    # Observation weights
    if isnothing(obs_weights)
        obs_weights = ones(length(obs_idx))
    end

    # Define ffun and hfun
    ffun_bar = ffun_bar_general
    hfun = hfun_general
    ffun(x, u, d, p, t, p_fixed) = ffun_bar(x, u(t), d(t), p, p_fixed)
    hfun_obs(x, u, d, p, p_fixed) = hfun(x, u, d, p, p_fixed)[obs_idx]

    # Fixed parameters
    p_fixed = (; NN_settings...,
                NN_settings, 
                hfun = hfun_obs,
                w_Ta = w_Ta,
                with_Q = with_Q,
                plant = plant,
                norm_params = norm_params,
                eval_Na_fun = eval_Na_fun,
                obs_idx = obs_idx,
                lambda_reg = lambda_reg,
                obs_weights = obs_weights,
                ode_solver_str = ode_solver_str
                )

    # Initialize parameters 
    #compute_ss_params = true
    p0, x0_guess_vec, θ0 = initialize_parameters(datasets, NN_settings, p_fixed, plant, w_Ta; compute_ss_params = compute_ss_params)

    # Add NN idxs to p_fixed
    p_fixed = (p_fixed..., 
            idx_NN = get_NN_idxs(NN_settings, θ0))

    
    # Create ODE problems for each dataset
    if isnothing(datasets)
        prob_list = nothing
    else
        prob_list = [make_ode_problem(ffun, ds, x0_guess_vec[i], p0, p_fixed) for (i, ds) in enumerate(datasets)];
    end            
    
    return ffun, ffun_bar, hfun, θ0, p_fixed, p0, x0_guess_vec, prob_list, NN_settings
      
end


