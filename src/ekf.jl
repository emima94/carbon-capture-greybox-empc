function ekf_update(hfun, x_prior, P_prior, u, dist, p, p_fixed, y_meas)

    # Get observation covariance
    Rv = p_fixed.Rv
    nx = p_fixed.nx

    # Get measurement prediction and Jacobian
    y_prior = hfun(x_prior, u, dist, p, p_fixed)

    H_k = ForwardDiff.jacobian(_x -> hfun(_x, u, dist, p, p_fixed), x_prior)

    # Innovation covariance
    R_prior = H_k * P_prior * H_k' + Rv

    # Innovation (residual)
    e_k = y_meas - y_prior

    # Kalman gain
    K_k = P_prior * H_k' * (R_prior \ I)  # equivalently: P_prior[k]*H_k'*inv(R_prior[k])

    # Updated state estimate
    x_post = x_prior + K_k * e_k

    # Updated state covariance
    I_n = Matrix{Float64}(I, nx, nx)
    P_post = (I_n - K_k * H_k) * P_prior

    # Updated observation (posterior)
    y_post = hfun(x_post, u, dist, p, p_fixed)

    R_post = H_k * P_post * H_k' + Rv

    return x_post, y_post, y_prior, P_post, R_post, R_prior

end


function moment_equations_EKF(x, u, dist, p, p_fixed, ffun)

    # Number of states
    nx = p_fixed.nx
    # Extract mean and covariance from x
    x_mean = x[1:nx]
    P = reshape(x[nx+1:end], (nx, nx))
    # Extract noise parameter from p_fixed
    sx = p_fixed.sx

    f = ffun(x_mean, u, dist, p, p_fixed)

    # Linearize f around current state to get A matrix (df/dx)
    A = ForwardDiff.jacobian(_x -> ffun(_x, u, dist, p, p_fixed), x_mean)
    #A = ffun_Jac(x_mean, u, dist, p, p_fixed)
    dPdt = A * P + P * A' + sx.^2 .* I(nx) # Process noise added to covariance dynamics

    return vcat(f, reshape(dPdt, :))

end


function ekf_predict(ffun, x0, P0, u, dist, p, p_fixed, t0, tf)

    nx = p_fixed.nx
    dt = p_fixed.dt

    x0P0 = vcat(x0, reshape(P0, :))

    t_vec = vcat(t0:dt:tf, tf)
    xP_prior = euler_simulation_ekf(ffun, x0P0, u, dist, p, p_fixed, t_vec)

    x_prior = xP_prior[1:nx]
    P_prior = reshape(xP_prior[nx+1:end], (nx, nx))

    return x_prior, P_prior
end



function euler_simulation_ekf(ffun, x0, u, dist, p, p_fixed, t_vec)
    nx = length(x0)
    
    x = x0

    for i in 1:(length(t_vec)-1)
        dt = t_vec[i+1] - t_vec[i]
        f = ffun(x, u[:,1], dist[:,1], p, p_fixed)
        x = x + dt * f
    end

    return x
end

function ekf_filter(ffun, hfun, x0, P0, u, dist, p, p_fixed, t_vec, y_meas)
"""
    ekf_filter(ODE_prob, obs_eq, p, x0, P0, t_vec, y_meas, Q, R)

Extended Kalman Filter: 1-step-ahead predictions and updates.

Inputs:
- `ffun`: function defining the system dynamics (state transition function)
- `hfun`: function defining the observation model
- `x0`: initial state estimate
- `P0`: initial state covariance
- `p`: model parameters (to be estimated)
- `p_fixed`: fixed model parameters and settings
- `t_vec`: vector of time points for filtering
- `y_meas`: matrix of measurements corresponding to `t_vec`

Returns:
    x_prior, x_post, y_prior, y_post, P_prior, P_post
"""
    
    n_steps     = length(t_vec)
    nx          = length(x0)
    ny          = size(y_meas, 1)
    Rv          = p_fixed.Rv
    dt          = p_fixed.dt

    # Preallocate arrays
    x_prior = zeros(nx, n_steps)
    x_post  = zeros(nx, n_steps)
    y_prior = zeros(ny,    n_steps)
    y_post  = zeros(ny,    n_steps)

    P_prior = Vector{Matrix{Float64}}(undef, n_steps)
    P_post  = Vector{Matrix{Float64}}(undef, n_steps)
    R_prior = Vector{Matrix{Float64}}(undef, n_steps)
    R_post  = Vector{Matrix{Float64}}(undef, n_steps)

    # Initial state
    x_post[:, 1] = x0
    y_post[:, 1] = hfun(x0, u[:,1], dist[:,1], p, p_fixed)
    P_post[1]    = P0
    H_k = ForwardDiff.jacobian(x -> hfun(x, u[:,1], dist[:,1], p, p_fixed), x0)
    R_post[1]    = H_k * P_post[1] * H_k' + Rv

    # Do update step for initial time step (k=1) to get x_post[:, 1], P_post[1], y_post[:, 1], R_post[1]
    #x_post[:, 1], y_post[:, 1], _, P_post[1], R_post[1], _ = ekf_update(hfun, x_prior[:, 1], P_prior[1], u[:,1], dist[:,1], p, p_fixed, y_meas[:,1])

    # Empty first entry to prior vectors
    x_prior[:, 1] = zeros(nx, 1) .* NaN
    y_prior[:, 1] = zeros(ny, 1) .* NaN
    P_prior[1]    = zeros(nx, nx) .* NaN
    R_prior[1]    = zeros(ny, ny) .* NaN

    for k in 2:n_steps

        t_prev = t_vec[k-1]
        t_cur  = t_vec[k]

        # # Prediction step
        x_prior_k, P_prior_k = ekf_predict(ffun, x_post[:, k-1], P_post[k-1], u[:,k-1], dist[:,k-1], p, p_fixed, t_prev, t_cur)

        x_prior[:, k] = x_prior_k
        P_prior[k] = P_prior_k

        # # Data update step (correction) ---
        x_post_k, y_post_k, y_prior_k, P_post_k, R_post_k, R_prior_k = ekf_update(hfun, x_prior[:, k], P_prior[k], u[:,k], dist[:,k], p, p_fixed, y_meas[:,k])

        x_post[:, k] = x_post_k
        y_post[:, k] = y_post_k
        y_prior[:, k] = y_prior_k
        P_post[k] = P_post_k
        R_post[k] = R_post_k
        R_prior[k] = R_prior_k


    end

    return x_prior, x_post, y_prior, y_post, P_prior, P_post, R_prior, R_post
    
end


function ekf_tk_ahead_prediction(t_vec_ekf, x_post, dist, settings, u = settings.u, time_control = settings.time_control)

    # Extract parameters and funcs from settings
    #ffun = settings.funcs.ffun_2du
    #hfun = settings.funcs.hfun_2du
    ffun = settings.funcs.ffun_model
    hfun = settings.funcs.hfun_model
    
    #u = settings.u
    #time_control = settings.time_control
    u_prev = settings.u_prev
    p_opt = settings.p_opt
    p_fixed = settings.p_fixed
    time_start = settings.time_start
    time_step_ahead = settings.time_step_ahead

    # Initialize before the loop
    t_pred = Vector{Float64}(undef, 0)
    x_pred = Matrix{Float64}(undef, size(x_post, 1), 0)
    y_pred = Matrix{Float64}(undef, length(hfun(x_post[:,1], u[1], dist, p_opt, p_fixed)), 0)  # adjust size to match y dimension

    # Make a k-step ahead prediction using the EKF posterior as the initial state estimate
    pred_idxs = 1:20:length(t_vec_ekf) # Loop over time points in t_vec_ekf (can adjust step size for less frequent predictions)
    for i in pred_idxs # Loop over time points in t_vec_ekf (can adjust step size for less frequent predictions)
        x0_i = x_post[:, i]
        t0_i = t_vec_ekf[i]
        tf_i = t0_i + time_step_ahead
        dt_sim = 10 / 3600 # Time step in Euler simulation for EKF prediction
        t_vec_i = t0_i:dt_sim:tf_i
        u_i = reduce(hcat, [get_control(t_vec_i[j], time_control, u, u_prev) for j in eachindex(t_vec_i)]) # Control input for EKF prediction at each time point in t_vec_i
        d_i = repeat(dist, 1, length(t_vec_i)) # Mean disturbance repeated for each time point in t_vec_i

        x_out = simulate_system(ffun, x0_i, u_i, d_i, p_opt, p_fixed, t_vec_i)
        y_out = [hfun(x_out[:, j], u_i[:, j], d_i[:, j], p_opt, p_fixed) for j in eachindex(t_vec_i)]
        y_out_mat = reduce(hcat, y_out)

        if i == 1
            t_pred = t_vec_i
            x_pred = x_out
            y_pred = y_out_mat
        else
            t_pred = vcat(t_pred, t_vec_i[end])
            x_pred = hcat(x_pred, x_out[:, end])
            y_pred = hcat(y_pred, y_out[end])
        end
    end

    time_pred =  Millisecond.(round.(Int, t_pred * 3600 * 1000)) + time_start

    return time_pred, x_pred, y_pred, t_pred

end

# function ekf_tk_ahead_prediction_general(ffun, hfun, x_post, u, dist, p, p_fixed, t_vec, time_step_ahead)
    
#     # Make a k-step ahead prediction using the EKF posterior as the initial state estimate
#     # Loop over time points in t_vec (can adjust step size for less frequent predictions)

#     # Initialize before the loop
#     t_pred = Vector{Float64}(undef, 0)
#     x_pred = Matrix{Float64}(undef, size(x_post, 1), 0)
#     y_pred = Matrix{Float64}(undef, length(hfun(x_post[:,1], u[:,1], dist[:,1], p, p_fixed)), 0)  # adjust size to match y dimension

#     # Find last idx in t_vec where t_vec[idx] <= t_vec[end] - time_step_ahead
#     last_idx = findlast(t -> t <= maximum([t_vec[1], t_vec[end] - time_step_ahead]), t_vec)

#     pred_idxs = 1:last_idx # Loop over time points in t_vec up to last_idx (can adjust step size for less frequent predictions)
  
#     for i in pred_idxs # Loop over time points in t_vec_ekf (can adjust step size for less frequent predictions)
#         x0_i = x_post[:, i]
#         t0_i = t_vec[i]
#         tf_i = minimum([t0_i + time_step_ahead, t_vec[end]])
#         Nx_per_step = 10 # Number of Euler steps per time step for EKF prediction

       
#         # Get idxs for prediction time points
#         idx_f_i = findfirst(tc -> tc >= tf_i, t_vec)
#         idx_i = i:idx_f_i
#         t_vec_i = t_vec[idx_i]

#         # Expand t_vec_i to have Nx_per_step points between each time step for smoother Euler simulation
#         t_vec_i_fine_grid = [t0_i; vcat([range(t_vec_i[j], t_vec_i[j+1], length = Nx_per_step + 1)[2:end] for j in 1:(length(t_vec_i)-1)]...)]
#         u_i = hcat(repeat(u[:, idx_i[1:end-1]], inner = (1,Nx_per_step)), u[:, idx_f_i]) # Control input for EKF prediction at each time point in t_vec_i_expanded
#         d_i = hcat(repeat(dist[:,idx_i[1:end-1]], inner = (1,Nx_per_step)), dist[:, idx_f_i]) # Disturbance for EKF prediction at each time point in t_vec_i_expanded

#         x_out = simulate_system(ffun, x0_i, u_i, d_i, p, p_fixed, t_vec_i_fine_grid)
#         y_out = [hfun(x_out[:, j], u_i[:, j], d_i[:, j], p, p_fixed) for j in eachindex(t_vec_i_fine_grid)]
#         y_out_mat = reduce(hcat, y_out)

#         if i == 1
#             t_pred = t_vec_i_fine_grid
#             x_pred = x_out
#             y_pred = y_out_mat
#         else
#             t_pred = vcat(t_pred, t_vec_i_fine_grid[end])
#             x_pred = hcat(x_pred, x_out[:, end])
#             y_pred = hcat(y_pred, y_out[end])
#         end
#     end

#     return x_pred, y_pred, t_pred

# end


function ekf_tk_ahead_prediction_general(ffun, hfun, x_post, u, dist, p, p_fixed, t_vec, time_step_ahead)
    
    # Make a k-step ahead prediction using the EKF posterior as the initial state estimate
    # Loop over time points in t_vec (can adjust step size for less frequent predictions)

    N = length(t_vec)
    nx = size(x_post, 1)
    ny = length(hfun(x_post[:,1], u[:,1], dist[:,1], p, p_fixed))

    # Allocate memoruy for predictions
    x_pred = Matrix{Float64}(undef, nx, N)
    y_pred = Matrix{Any}(undef, ny, N)  # adjust size to match y dimension
    
    # Initialize before the loop
    x_pred[:, 1] = x_post[:, 1]
    y_pred[:, 1] = hfun(x_post[:,1], u[:,1], dist[:,1], p, p_fixed)

    Nx_per_step = 10 # Number of Euler steps per time step for EKF prediction

    for i_final in 2:N # Loop over time points in t_vec_ekf (can adjust step size for less frequent predictions)

        tf_i = t_vec[i_final]

        if tf_i <= time_step_ahead
            i_start = i_final - 1
            x0_i = x_pred[:, i_start]
        else
            i_start = min(argmin(abs.(t_vec .- (tf_i - time_step_ahead))), i_final - 1) # Find index of time point in t_vec closest to tf_i - time_step_ahead, but not greater than i_final - 1
            x0_i = x_post[:, i_start]
        end
       
        t0_i = t_vec[i_start]

        # Get idxs for prediction time points
        t_vec_i = t_vec[i_start:i_final]

        # Expand t_vec_i to have Nx_per_step points between each time step for smoother Euler simulation
        t_vec_i_fine_grid = [t0_i; vcat([range(t_vec_i[j], t_vec_i[j+1], length = Nx_per_step + 1)[2:end] for j in 1:(length(t_vec_i)-1)]...)]
        u_i = hcat(repeat(u[:, i_start:i_final-1], inner = (1,Nx_per_step)), u[:, i_final]) # Control input for EKF prediction at each time point in t_vec_i_expanded
        d_i = hcat(repeat(dist[:,i_start:i_final-1], inner = (1,Nx_per_step)), dist[:, i_final]) # Disturbance for EKF prediction at each time point in t_vec_i_expanded

        x_out = simulate_system(ffun, x0_i, u_i, d_i, p, p_fixed, t_vec_i_fine_grid)
        y_out = [hfun(x_out[:, j], u_i[:, j], d_i[:, j], p, p_fixed) for j in eachindex(t_vec_i_fine_grid)]
        y_out_mat = reduce(hcat, y_out)

        x_pred[:, i_final] = x_out[:, end]
        y_pred[:, i_final] = y_out_mat[:, end]

    end

    return x_pred, y_pred

end

function get_EKF_noise_params(plant, camp_group, nx, ny)
    # Choose EKF noise parameters based on model and data campaign
    if plant == "Tiller"
        sx = 0.2 * ones(nx)
        Rv = 0.75 .* I(ny)
        Rv_adap = Rv
        sx_adaptive = [1.0]
        #sx_adaptive = [0.1, 1.0]
    elseif plant == "DTI" && (camp_group == "camp_1_2_3" || camp_group == "camp_1_2_3_4")
        sx = 0.2 * ones(nx)
        Rv = [1.0, 1.0, 1.0, 5.0] .^2 .* I(ny)
        Rv_adap = [1.0, 1.0, 1.0, 5.0] .^2 .* I(ny)
        sx_adaptive = 0.4 .* [0.5, 0.5]
        #sx_adaptive = 1.0 .* [0.1, 0.5, 0.1, 0.5]
    elseif plant == "DTI" && camp_group == "camp_4"
        sx = 0.2 * ones(nx)
        # Rv = [1.0, 1.0, 5.0] .^2 .* I(ny)
        # Rv_adap = [1.0, 1.0, 5.0] .^2 .* I(ny) # Higher measurement noise for Q in adaptive model
        Rv = [1.0, 1.0, 5.0] .^2 .* I(ny)
        Rv_adap = [1.0, 1.0, 5.0] .^2 .* I(ny) # Higher measurement noise for Q in adaptive model
        #Rv_adap = [1.0, 1.0, 5.0] .^2 .* I(ny) # Same measurement noise for Q in adaptive model (since Q is less noisy in camp 4)
        
        sx_adaptive = 0.4 .* [0.5, 0.5]
        # sx_adaptive = 1.0 .* [0.1, 0.5, 0.1, 0.5]
    else
        error("Unknown EKF noise settings for plant $(plant) and camp group $(camp_group). Please specify sx, Rv, Rv_adap, and sx_adaptive for this case in get_EKF_noise_params function.")
    end

    return sx, Rv, Rv_adap, sx_adaptive
end