function generate_JuMP_model_ref(p_scenario, p)

    # Extract parameters and functions
    cap_eff_ref = p_scenario.cap_eff_ref
    lambda_F = p_scenario.lambda
    x0 = p_scenario.x0
    F_min = p_scenario.lb
    F_max = p_scenario.ub
    N = p_scenario.N
    Nx = p_scenario.Nx
    interval_of = p_scenario.interval_of
    D = p_scenario.D
    dt = p_scenario.dt
    dt_sim = p_scenario.dt_sim
    ffun = p_scenario.ffun
    F_prev = p_scenario.F_prev
    Q = p_scenario.Q
    hfun = p_scenario.hfun
    ode_solver = p_scenario.ODE_solver

    # ----- Model -------
    model = Model(Ipopt.Optimizer)
    set_attribute(model, "print_level", 5)

    # Initial state
    nx = length(x0)
    nz = length(hfun(x0, fill(0.0, 2), D[:,1], p))

    # States (fine grid, Nx nodes)
    @variable(model, x[1:nx, 0:(Nx-1)]) # 5 states: ca, cd, ct, Ma, Mina
    # Set initial guess for states (constant at initial state)
    X_guess = repeat(x0, 1, Nx) # Initial guess for states (constant at initial state)
    set_start_value.(x, X_guess)



    # Controls (coarse grid, N intervals)
    @variable(model, F[1:N], lower_bound=F_min, upper_bound=F_max)

    # Initial guess for controls
    F_guess = fill(0.3, N)
    set_start_value.(F, F_guess)

    # constraints
    @constraint(model, initial_state[i = 1:nx],  x[i,0] == x0[i]) # Initial condition

    # ODE solver
    if ode_solver == "euler"
        # dynamic constraints using Euler collocation
        for i in 1:Nx-1
            # Determine which control interval this state belongs to
            k = interval_of[i] # control interval index for state node i-1
            # Evaluate dynamics at current state and control
            u_k = [F[k], Q[k]]
            d_k = D[:,k]
            f = ffun(x[:,i-1], u_k, d_k, p)
            for s in 1:nx 
                @constraint(model, x[s,i] == x[s,i-1] + dt_sim * f[s])
            end
        end
    elseif ode_solver == "rk4"
        # Dynamic constraints using RK4 collocation with intermediate variables
           # # Intermediate RK4 variables for collocation
        @variable(model, _k1[1:nx, 1:(Nx-1)])
        @variable(model, _k2[1:nx, 1:(Nx-1)])
        @variable(model, _k3[1:nx, 1:(Nx-1)])
        @variable(model, _k4[1:nx, 1:(Nx-1)])
        set_start_value.(_k1, X_guess[:,1:end-1])
        set_start_value.(_k2, X_guess[:,1:end-1])
        set_start_value.(_k3, X_guess[:,1:end-1])
        set_start_value.(_k4, X_guess[:,1:end-1])

        # Dynamic constraints using RK4 collocation with intermediate variables
        for i in 1:Nx-1
            k = interval_of[i] # control interval index for state node i-1
            u_k = [F[k], Q[k]]
            d_k = D[:,k]
            x_i = x[:,i-1]

            @constraint(model, _k1[:,i] == ffun(x_i, u_k, d_k, p))
            @constraint(model, _k2[:,i] == ffun(x_i + dt_sim/2.0 * _k1[:,i], u_k, d_k, p))
            @constraint(model, _k3[:,i] == ffun(x_i + dt_sim/2.0 * _k2[:,i], u_k, d_k, p))
            @constraint(model, _k4[:,i] == ffun(x_i + dt_sim * _k3[:,i], u_k, d_k, p))
            @constraint(model, x[:,i] == x_i + dt_sim/6.0 * (_k1[:,i] + 2*_k2[:,i] + 2*_k3[:,i] + _k4[:,i]))
        end
    else
        error("Unsupported ODE solver: $ode_solver")
    end


    # Get capture efficiency at each time step
    @expression(model, cap_eff[j=1:Nx-1], 
                    hfun(x[:,j], 
                        [F[interval_of[j]], Q[interval_of[j]]], 
                        D[:,interval_of[j]], 
                        p)[3]
    )

    # Objective: 
    # Reference control of capture efficiency.
    @objective(model, Min,
        sum((cap_eff[j] - cap_eff_ref[j])^2 for j in 1:Nx-1) * dt_sim
            + lambda_F * ((F[1] - F_prev)^2 + sum((F[k] - F[k-1])^2 for k = 2:N)) 
    )

    return model

end

function generate_JuMP_model_EMPC(p_scenario, p)


    # Extract parameters and functions
    p_el = p_scenario.p_el
    p_CO2 = p_scenario.p_CO2
    lambda = p_scenario.lambda
    eta_min = p_scenario.eta_min
    x0 = p_scenario.x0
    lb = p_scenario.lb
    ub = p_scenario.ub
    N = p_scenario.N
    Nx = p_scenario.Nx
    interval_of = p_scenario.interval_of
    D = p_scenario.D
    dt = p_scenario.dt
    dt_sim = p_scenario.dt_sim
    t = p_scenario.t
    t_sim = p_scenario.t_sim
    ffun = p_scenario.ffun
    hfun = p_scenario.hfun
    F_prev = p_scenario.F_prev
    Q_prev = p_scenario.Q_prev
    
    lambda_F, lambda_Q, lambda_s = lambda

    # ----- Model -------
    model = Model(Ipopt.Optimizer)
    set_attribute(model, "print_level", 5)

    # Initial state
    
    nx = length(x0)

    # States (fine grid, Nx nodes)
    @variable(model, x[1:nx, 0:(Nx-1)]) # 5 states: ca, cd, ct, Ma, Mina
    # Set initial guess for states (constant at initial state)
    X_guess = repeat(x0, 1, Nx) # Initial guess for states (constant at initial state)
    set_start_value.(x, X_guess)

    # Controls (coarse grid, N intervals)
    # bounds
    F_min, Q_min, s_min = lb
    F_max, Q_max, s_max = ub

    @variable(model, F[1:N], lower_bound=F_min, upper_bound=F_max)
    @variable(model, Q[1:N], lower_bound=Q_min, upper_bound=Q_max)
    @variable(model, s, lower_bound=s_min) # Slack variable for constraint violation

    # Initial guess for controls
    F_guess = fill(0.3, N)
    Q_guess = fill(28.0, N)
    set_start_value.(F, F_guess)
    set_start_value.(Q, Q_guess)
    set_start_value(s, 0.0)

    # constraints
    @constraint(model, initial_state[i = 1:nx],  x[i,0] == x0[i]) # Initial condition

    # dynamic constraints using Euler collocation
    for i in 1:Nx-1
        # Determine which control interval this state belongs to
        k = interval_of[i] # control interval index for state node i-1
        # Evaluate dynamics at current state and control
        u_k = [F[k], Q[k]]
        d_k = D[:,k]
        f = ffun(x[:,i-1], u_k, d_k, p)
        for s in 1:nx 
            @constraint(model, x[s,i] == x[s,i-1] + dt_sim * f[s])
        end
    end

    # Output constraint at final time step (Ensure capture efficiency meets minimum requirement at average over the horizon)
    @constraint(model, x[4,end] - x[5,end] * eta_min + s >= 0); # Ma(tf) - Mina(tf)*eta_min/100 + s >= 0

    # Objective: 
    # Assuming a constant CO2 price 
    @objective(model, Min,
        sum(p_el[k] * Q[k] * dt for k = 1:N) 
            + p_CO2 * x[4,end] 
            + lambda_F * ((F[1] - F_prev)^2 + sum((F[k] - F[k-1])^2 for k = 2:N)) 
            + lambda_Q * ((Q[1] - Q_prev)^2 + sum((Q[k] - Q[k-1])^2 for k = 2:N)) 
            + lambda_s * s^2
    )

    return model

end

function generate_JuMP_model_EMPC_no_price(p_scenario, p)
    

    # Extract parameters and functions
    p_el = p_scenario.p_el
    p_CO2 = p_scenario.p_CO2
    lambda = p_scenario.lambda
    eta_min = p_scenario.eta_min
    x0 = p_scenario.x0
    lb = p_scenario.lb
    ub = p_scenario.ub
    N = p_scenario.N
    Nx = p_scenario.Nx
    interval_of = p_scenario.interval_of
    D = p_scenario.D
    dt = p_scenario.dt
    dt_sim = p_scenario.dt_sim
    t = p_scenario.t
    t_sim = p_scenario.t_sim
    ffun = p_scenario.ffun
    hfun = p_scenario.hfun
    F_prev = p_scenario.F_prev
    Q_prev = p_scenario.Q_prev
    
    lambda_F, lambda_Q, lambda_s = lambda

    model = generate_JuMP_model_EMPC(p_scenario, p)

    Q = model[:Q]
    F = model[:F]
    s = model[:s]

    # Rewrite objective to only minimize electricity consumption without considering price
    @objective(model, Min,
        sum(Q[k]^2 for k = 1:N) * dt
            + lambda_F * ((F[1] - F_prev)^2 + sum((F[k] - F[k-1])^2 for k = 2:N)) 
            + lambda_Q * ((Q[1] - Q_prev)^2 + sum((Q[k] - Q[k-1])^2 for k = 2:N)) 
            + lambda_s * s^2
    )

    return model
end



function generate_JuMP_model_ref_SS(p_scenario, p, p_fixed)

    " 
    A JuMP model for steady state optimization, to compute the optimal 
        steady state operating point for a given capture efficiency reference.

    Optimization variables:
    - x_ss: Steady state values of the states 
    - F_ss: Steady state value of the control input F
    If Tiller:
    - Q_ss: Steady state value of the control input Q
    If DTI:
    - TlD_HX_ss: Steady state value of the control input TlD_HX_SP

    Optimization problem:
    min_{x, u} phi = lambda_Q * Q^2

    s.t.

    - Steady state constraints: f(x_ss, u_ss, d, p) = 0
    - Capture efficiency constraint: h_cap_eff(x_ss, u_ss, d, p) = cap_eff_ref
    - Control input bounds: F_min <= F_ss <= F_max, Q_min <= Q_ss <= Q_max (if Tiller), TlD_HX_min <= TlD_HX_ss <= TlD_HX_max (if DTI)
    "

    # Extract parameters and functions
    cap_eff_ref = p_scenario.cap_eff_ref
    x_ss_guess = p_scenario.x_ss_guess
    u_ss_guess = p_scenario.u_ss_guess
    lb = p_scenario.lb
    ub = p_scenario.ub
    ffun = p_scenario.ffun
    hfun_cap_eff = p_scenario.hfun_cap_eff
    hfun_Q = p_scenario.hfun_Q
    dist = p_scenario.dist
    lambda = p_scenario.lambda

    lambda_cap_eff, lambda_Q = lambda

    # ----- Model -------
    model = Model(Ipopt.Optimizer)
    #set_attribute(model, "print_level", 5)
    set_silent(model)

    # Steady state optimization variables
    nx = length(x_ss_guess)
    nu = length(u_ss_guess)
    @variable(model, x_ss[1:nx]) # Steady state values of the states
    @variable(model, u_ss[i=1:nu], lower_bound=lb[i], upper_bound=ub[i]) # Steady state values of the control inputs
    
    # Initial guess for optimization variables
    set_start_value.(x_ss, x_ss_guess)
    set_start_value.(u_ss, u_ss_guess)

    # Steady state constraints: f(x_ss, u_ss, d, p) = 0
    @constraint(model, steady_state_con[i=1:nx], ffun(x_ss, u_ss, dist, p, p_fixed)[i] == 0)

    # Get steady state capture efficiency from hfun
    @expression(model, cap_eff_ss, hfun_cap_eff(x_ss, u_ss, dist, p, p_fixed))

    # Capture efficiency constraint: cap_eff_ss == cap_eff_ref
    @constraint(model, cap_eff_constraint, cap_eff_ss == cap_eff_ref)

    @expression(model, reboiler_duty_ss, hfun_Q(x_ss, u_ss, dist, p, p_fixed))

    # Objective: 
    # Reference control of capture efficiency and minimization of reboiler duty.
    @objective(model, Min,
        lambda_Q * reboiler_duty_ss^2
    )

    return model

end



function generate_JuMP_model_EMPC_DTI(p_scenario, p, p_fixed, model_type)


    # Extract parameters and functions
    p_el = p_scenario.p_el
    lambda = p_scenario.lambda
    eta_min = p_scenario.eta_min
    x0 = p_scenario.x0
    lb = p_scenario.lb
    ub = p_scenario.ub
    N = p_scenario.N    # Number of control intervals
    Nx = p_scenario.Nx  # Number of time steps for simulation
    dist = p_scenario.dist
    dt = p_scenario.dt
    dt_sim = p_scenario.dt_sim # Time step for simulation
    ffun = p_scenario.ffun # System dynamics function x: ca, cd, TlD_Hx, Ma, Mina. u: F, TlD_HX_SP. d: cgina, Fgina.
    hfun_Q = p_scenario.hfun_Q # Output function for reboiler duty Q
    u_prev = p_scenario.u_prev # Previous value of u (for penalizing changes in the first control interval) Set to optimal steady state value
    du = p_scenario.du # Minimum change in control input per interval
    

    interval_of = repeat(1:N, inner=Int((Nx-1)/N)) # Vector mapping each state time step to the corresponding control interval index

    lambda_u, lambda_s_eta_min, lambda_s_x = lambda
    du_min, du_max = du

    nx = length(x0)
    nu = length(u_prev)
    nd = size(dist, 1)

    # ----- Model -------
    model = Model(Ipopt.Optimizer)
    set_attribute(model, "print_level", 5)

    # States (fine grid, Nx nodes)
    @variable(model, x[1:nx, 0:(Nx-1)]) # 5 states: ca, cd, TlD_HX, Ma, Mina
    # Set initial guess for states (constant at initial state)
    X_guess = repeat(x0, 1, Nx) # Initial guess for states (constant at initial state)
    set_start_value.(x, X_guess)

    # Controls (coarse grid, N intervals)
    # bounds
    u_min, s_eta_min_min, s_x_min = lb
    u_max, s_eta_min_max, s_x_max = ub

    @variable(model, u[i = 1:nu, j = 1:N], lower_bound=u_min[i], upper_bound=u_max[i]) # Control inputs F and TlD_HX_SP
    @variable(model, s_eta_min, lower_bound = s_eta_min_min) # Slack variable for constraint violation
    @variable(model, s_x1, lower_bound = s_x_min) # Slack variable for constraint violation
    @variable(model, s_x2, lower_bound = s_x_min) # Slack variable for constraint violation

    # Initial guess for controls

    set_start_value.(u, repeat(u_prev, 1, N))
    set_start_value(s_eta_min, 0.0)
    set_start_value(s_x1, 0.0)
    set_start_value(s_x2, 0.0)

    # constraints
    @constraint(model, initial_state[i = 1:nx],  x[i,0] == x0[i]) # Initial condition

    # dynamic constraints using Euler collocation
    for j in 1:Nx-1
        # Evaluate dynamics at current state and control
        x_j = [x[i,j-1] for i in 1:nx]
        u_j = u[:,interval_of[j]]
        d_j = dist[:,j]
        f = ffun(x_j, [u_j[1]; NaN; u_j[2]], d_j, p, p_fixed)
        for i in 1:nx 
            @constraint(model, x[i,j] == x[i,j-1] + dt_sim * f[i])
        end
    end

    # Output constraint at final time step (Ensure capture efficiency meets minimum requirement at average over the horizon)
    @constraint(model, x[4,end] - x[5,end] * eta_min/100 + s_eta_min >= 0); # Ma(tf) - Mina(tf)*eta_min/100 + s_eta_min >= 0

    # Constraint on final CO2 concentration in desorber (to avoid using capture capacity for the next horizon)
    @constraint(model, x[1,0] == x[1,end] + s_x1) # ca
    @constraint(model, x[2,0] == x[2,end] + s_x2) # cd

    #  # These work well for linear model:
    #@constraint(model, x[1,end] + s_x1 >= x[1,0]) # ca
    #@constraint(model, x[2,0] <= x[2,end] + s_x2) # cd
#    @constraint(model, (x[1,0] - x[2,0]) - (x[1,end] - x[2,end]) + s_x1 + s_x2 >= 0) # ca - cd

    # Input change constraint
    # First control interval:
    for i in 1:nu
        @constraint(model, du_min[i] <= (u[i,1] - u_prev[i])/dt <= du_max[i])
    end
    # Subsequent control intervals:
    for k in 2:N
        for i in 1:nu
             @constraint(model, du_min[i] <= (u[i,k] - u[i,k-1])/dt <= du_max[i])
        end
    end

    # Get Q (reboiler duty) at each time step from hfun_Q
    Q = [hfun_Q([x[i,j] for i in 1:nx], u[:,interval_of[j]], dist[:,interval_of[j]], p, p_fixed) for j in 1:Nx-1]

    # Objective: 
    @objective(model, Min,
        sum(p_el[j] * Q[j] * dt_sim for j = 1:Nx-1) 
            + lambda_u[1] * ((u[1,1] - u_prev[1])^2 + sum((u[1,k] - u[1,k-1])^2 for k = 2:N)) 
            + lambda_u[2] * ((u[2,1] - u_prev[2])^2 + sum((u[2,k] - u[2,k-1])^2 for k = 2:N)) 
            + lambda_s_eta_min * s_eta_min^2
            + lambda_s_x * s_x1^2
            + lambda_s_x * s_x2^2
    )

    return model

end




function generate_JuMP_model_EMPC_DTI_closed_loop(p_scenario, p, p_fixed)
""" 
    This function generates a JuMP model for EMPC, used in closed-loop simulations. 

    Inputs:
    p_scenario:     Scenario specific parameters and functions, including:
                - p_el: Electricity price vector over the horizon
                - lambda: Weights for the objective function
                - eta_min: Minimum average capture efficiency requirement (in %)
                - x0: Initial state at the beginning of the horizon
                - lb: Lower bounds for control inputs and slack variables
                - ub: Upper bounds for control inputs and slack variables
                - N: Number of control intervals in the horizon
                - Nx: Number of time steps for simulation in the horizon
                - dist: Disturbance scenario (cgina, Fgina) over the horizon
                - dt: Time step for control input changes
                - dt_sim: Time step for system simulation (should be <= dt)
                - ffun: System dynamics function
                - hfun_Q: Output function to compute reboiler duty Q from states and controls
                - u_prev: Control input values from the previous time step (for penalizing changes in the first control interval)
                - du: Minimum and maximum change in control input per interval (for input rate constraints)
    
    p:             Model parameters (estimated) that are passed to the dynamics and output functions
    p_fixed:       Fixed parameters that are passed to the dynamics and output functions 

"""

    # Extract parameters and functions
    p_el = p_scenario.p_el
    lambda = p_scenario.lambda
    eta_min = p_scenario.eta_min
    x0 = p_scenario.x0
    lb = p_scenario.lb
    ub = p_scenario.ub
    N = p_scenario.N    # Number of control intervals
    Nx = p_scenario.Nx  # Number of time steps for simulation, in total, Nx >= N
    dist = p_scenario.dist
    dt = p_scenario.dt  # Time step for control interval
    dt_sim = p_scenario.dt_sim # Time step for simulation
    ffun = p_scenario.ffun # System dynamics function x: ca, cd, TlD_Hx, Ma, Mina. u: F, TlD_HX_SP. d: cgina, Fgina.
    hfun_Q = p_scenario.hfun_Q # Output function for reboiler duty Q
    u_prev = p_scenario.u_prev # Previous value of u (for penalizing changes in the first control interval) Set to optimal steady state value
    du = p_scenario.du # Minimum change in control input per interval
    control_interval_start_horizon2 = p_scenario.control_interval_start_horizon2 # Control interval index for the first time step in the second horizon.
    
    # Compute the mapping from state time steps to control intervals
    interval_of = repeat(1:N, inner=Int((Nx-1)/N)) # Vector mapping each state time step to the corresponding control interval index

    lambda_u, lambda_s_eta_min, lambda_s_x = lambda
    du_min, du_max = du

    nx = length(x0)
    nu = length(u_prev)
    nd = size(dist, 1)

    # ----- Model -------
    model = Model(Ipopt.Optimizer)
    #set_attribute(model, "print_level", 5)
    set_silent(model)

    # States (fine grid, Nx nodes)
    @variable(model, x[1:nx, 0:(Nx-1)]) # 5 states: ca, cd, TlD_HX, Ma, Mina
    # Set initial guess for states (constant at initial state)
    X_guess = repeat(x0, 1, Nx) # Initial guess for states (constant at initial state)
    set_start_value.(x, X_guess)

    # Controls (coarse grid, N intervals)
    # bounds
    u_min, s_eta_min_min, s_x_min = lb
    u_max, s_eta_min_max, s_x_max = ub

    @variable(model, u[i = 1:nu, j = 1:N], lower_bound=u_min[i], upper_bound=u_max[i]) # Control inputs F and TlD_HX_SP
    @variable(model, s_eta_min_1, lower_bound = s_eta_min_min) # Slack variable for constraint violation
    @variable(model, s_eta_min_2, lower_bound = s_eta_min_min) # Slack variable for constraint violation
    @variable(model, s_x1, lower_bound = s_x_min) # Slack variable for constraint violation
    @variable(model, s_x2, lower_bound = s_x_min) # Slack variable for constraint violation

    # Initial guess for controls

    set_start_value.(u, repeat(u_prev, 1, N))
    set_start_value(s_eta_min_1, 0.0)
    set_start_value(s_eta_min_2, 0.0)
    set_start_value(s_x1, 0.0)
    set_start_value(s_x2, 0.0)

    # constraints
    @constraint(model, initial_state[i = 1:nx],  x[i,0] == x0[i]) # Initial condition

    # dynamic constraints using Euler collocation
    for j in 1:Nx-1
        # Evaluate dynamics at current state and control
        x_j = [x[i,j-1] for i in 1:nx]
        u_j = u[:,interval_of[j]]
        d_j = dist[:,j]
        f = ffun(x_j, u_j, d_j, p, p_fixed)
        for i in 1:nx 
            @constraint(model, x[i,j] == x[i,j-1] + dt_sim * f[i])
        end
    end

    # Test ffun
    # Output constraints: Ensure capture efficiency meets minimum requirement at average over both horizon 1 and 2
    idx_start_horizon2 = Int((Nx-1)/N) * (control_interval_start_horizon2-1) # Index of the first time step in horizon 2
    Ma_0_horizon2 = x[end-1,idx_start_horizon2] # Ma at the first time step of horizon 2
    Mina_0_horizon2 = x[end,idx_start_horizon2] # Mina at the first time step of horizon 2
    Ma_f_horizon2 = x[end-1,end] - Ma_0_horizon2 # Change in Ma over horizon 2
    Mina_f_horizon2 = x[end,end] - Mina_0_horizon2 # Change in Mina over horizon 2

    # Horizon 1:
    @constraint(model, Ma_0_horizon2 - Mina_0_horizon2 * eta_min/100 + s_eta_min_1 >= 0); 
    # Horizon 2:
    @constraint(model, Ma_f_horizon2 - Mina_f_horizon2 * eta_min/100 + s_eta_min_2 >= 0); 

    # Constraint on final CO2 concentration in absorber and desorber (to avoid using capture capacity for the next horizon)
    @constraint(model, x[1,0] == x[1,end] + s_x1) # ca
    @constraint(model, x[2,0] == x[2,end] + s_x2) # cd

    # Input change constraint
    # First control interval:
    for i in 1:nu
        @constraint(model, du_min[i] <= (u[i,1] - u_prev[i])/dt <= du_max[i])
    end
    # Subsequent control intervals:
    for k in 2:N
        for i in 1:nu
             @constraint(model, du_min[i] <= (u[i,k] - u[i,k-1])/dt <= du_max[i])
        end
    end

    # Get Q (reboiler duty) at each time step from hfun_Q
    Q = [hfun_Q([x[i,j] for i in 1:nx], u[:,interval_of[j]], dist[:,interval_of[j]], p, p_fixed) for j in 1:Nx-1]

    # Objective: 
    @objective(model, Min,
        sum(p_el[j] * Q[j] * dt_sim for j = 1:Nx-1) 
            + lambda_u[1] * ((u[1,1] - u_prev[1])^2 + sum((u[1,k] - u[1,k-1])^2 for k = 2:N)) 
            + lambda_u[2] * ((u[2,1] - u_prev[2])^2 + sum((u[2,k] - u[2,k-1])^2 for k = 2:N)) 
            + lambda_s_eta_min * s_eta_min_1^2
            + lambda_s_eta_min * s_eta_min_2^2
            + lambda_s_x * s_x1^2
            + lambda_s_x * s_x2^2
    )

    return model

end




function generate_JuMP_model_ref_MPC_DTI_closed_loop(p_scenario, p, p_fixed)
""" 
    This function generates a JuMP model for reference MPC, used in closed-loop simulations. 
        Minimization of energy consumption (reboiler duty) while tracking a reference capture efficiency, without considering electricity price.

    Inputs:
    p_scenario:     Scenario specific parameters and functions, including:
                - lambda: Weights for the objective function
                - eta_min: Minimum average capture efficiency requirement (in %)
                - x0: Initial state at the beginning of the horizon
                - lb: Lower bounds for control inputs and slack variables
                - ub: Upper bounds for control inputs and slack variables
                - N: Number of control intervals in the horizon
                - Nx: Number of time steps for simulation in the horizon
                - dist: Disturbance scenario (cgina, Fgina) over the horizon
                - dt: Time step for control input changes
                - dt_sim: Time step for system simulation (should be <= dt)
                - ffun: System dynamics function
                - hfun_cap_eff: Output function to compute capture efficiency from states and controls
                - hfun_Q: Output function to compute reboiler duty Q from states and controls
                - u_prev: Control input values from the previous time step (for penalizing changes in the first control interval)
                - du: Minimum and maximum change in control input per interval (for input rate constraints)
    
    p:             Model parameters (estimated) that are passed to the dynamics and output functions
    p_fixed:       Fixed parameters that are passed to the dynamics and output functions 

"""

    # Extract parameters and functions
    #p_el = p_scenario.p_el
    lambda = p_scenario.lambda
    eta_min = p_scenario.eta_min
    x0 = p_scenario.x0
    lb = p_scenario.lb
    ub = p_scenario.ub
    N = p_scenario.N    # Number of control intervals
    Nx = p_scenario.Nx  # Number of time steps for simulation, in total, Nx >= N
    dist = p_scenario.dist
    dt = p_scenario.dt  # Time step for control interval
    dt_sim = p_scenario.dt_sim # Time step for simulation
    ffun = p_scenario.ffun # System dynamics function x: ca, cd, TlD_Hx, Ma, Mina. u: F, TlD_HX_SP. d: cgina, Fgina.
    hfun_Q = p_scenario.hfun_Q # Output function for reboiler duty Q
    hfun_cap_eff = p_scenario.hfun_cap_eff # Output function for capture efficiency
    u_prev = p_scenario.u_prev # Previous value of u (for penalizing changes in the first control interval) Set to optimal steady state value
    du = p_scenario.du # Minimum change in control input per interval
    control_interval_start_horizon2 = p_scenario.control_interval_start_horizon2 # Control interval index for the first time step in the second horizon.
    
    # Compute the mapping from state time steps to control intervals
    interval_of = repeat(1:N, inner=Int((Nx-1)/N)) # Vector mapping each state time step to the corresponding control interval index

    #lambda_u, lambda_eta_min, lambda_Q, lambda_s_x = lambda
    lambda_u, lambda_Q, lambda_s_eta_min, lambda_s_x = lambda
    du_min, du_max = du

    nx = length(x0)
    nu = length(u_prev)
    nd = size(dist, 1)

    # ----- Model -------
    model = Model(Ipopt.Optimizer)
    #set_attribute(model, "print_level", 5)
    set_silent(model)

    # States (fine grid, Nx nodes)
    @variable(model, x[1:nx, 0:(Nx-1)]) # 5 states: ca, cd, TlD_HX, Ma, Mina
    # Set initial guess for states (constant at initial state)
    X_guess = repeat(x0, 1, Nx) # Initial guess for states (constant at initial state)
    set_start_value.(x, X_guess)

    # Controls (coarse grid, N intervals)
    # bounds
    u_min, s_eta_min_min, s_x_min = lb
    u_max, s_eta_min_max, s_x_max = ub

    @variable(model, u[i = 1:nu, j = 1:N], lower_bound=u_min[i], upper_bound=u_max[i]) # Control inputs F and TlD_HX_SP
    @variable(model, s_eta_min_1, lower_bound = s_eta_min_min) # Slack variable for constraint violation
    @variable(model, s_eta_min_2, lower_bound = s_eta_min_min) # Slack variable for constraint violation
    @variable(model, s_x1, lower_bound = s_x_min) # Slack variable for constraint violation
    @variable(model, s_x2, lower_bound = s_x_min) # Slack variable for constraint violation

    # Initial guess for controls

    set_start_value.(u, repeat(u_prev, 1, N))
    set_start_value(s_eta_min_1, 0.0)
    set_start_value(s_eta_min_2, 0.0)
    set_start_value(s_x1, 0.0)
    set_start_value(s_x2, 0.0)

    # constraints
    @constraint(model, initial_state[i = 1:nx],  x[i,0] == x0[i]) # Initial condition

    # dynamic constraints using Euler collocation
    for j in 1:Nx-1
        # Evaluate dynamics at current state and control
        x_j = [x[i,j-1] for i in 1:nx]
        u_j = u[:,interval_of[j]]
        d_j = dist[:,j]
        f = ffun(x_j, u_j, d_j, p, p_fixed)
        for i in 1:nx 
            @constraint(model, x[i,j] == x[i,j-1] + dt_sim * f[i])
        end
    end

    # Output constraints: Ensure capture efficiency meets minimum requirement at average over both horizon 1 and 2
    idx_start_horizon2 = Int((Nx-1)/N) * (control_interval_start_horizon2-1) # Index of the first time step in horizon 2
    Ma_0_horizon2 = x[4,idx_start_horizon2] # Ma at the first time step of horizon 2
    Mina_0_horizon2 = x[5,idx_start_horizon2] # Mina at the first time step of horizon 2
    Ma_f_horizon2 = x[4,end] - Ma_0_horizon2 # Change in Ma over horizon 2
    Mina_f_horizon2 = x[5,end] - Mina_0_horizon2 # Change in Mina over horizon 2

    # Horizon 1:
    @constraint(model, Ma_0_horizon2 - Mina_0_horizon2 * eta_min/100 + s_eta_min_1 >= 0); 
    # Horizon 2:
    @constraint(model, Ma_f_horizon2 - Mina_f_horizon2 * eta_min/100 + s_eta_min_2 >= 0); 

    # Constraint on final CO2 concentration in absorber and desorber (to avoid using capture capacity for the next horizon)
    @constraint(model, x[1,0] == x[1,end] + s_x1) # ca
    @constraint(model, x[2,0] == x[2,end] + s_x2) # cd

    # Input change constraint
    # First control interval:
    for i in 1:nu
        @constraint(model, du_min[i] <= (u[i,1] - u_prev[i])/dt <= du_max[i])
    end
    # Subsequent control intervals:
    for k in 2:N
        for i in 1:nu
             @constraint(model, du_min[i] <= (u[i,k] - u[i,k-1])/dt <= du_max[i])
        end
    end

    # Get Q (reboiler duty) at each time step from hfun_Q
    Q = [hfun_Q([x[i,j] for i in 1:nx], u[:,interval_of[j]], dist[:,interval_of[j]], p, p_fixed) for j in 1:Nx-1]

    # Get cap eff at each time step from hfun_cap_eff
    #cap_eff = [hfun_cap_eff([x[i,j] for i in 1:nx], u[:,interval_of[j]], dist[:,interval_of[j]], p, p_fixed) for j in 1:Nx-1]

    # Objective: 
    @objective(model, Min,
        #lambda_eta_min * sum((cap_eff[j] - eta_min)^2 for j in 1:Nx-1) * dt_sim + 
        lambda_Q * sum(Q[j] for j = 1:Nx-1) * dt_sim + 
        lambda_u[1] * ((u[1,1] - u_prev[1])^2 + sum((u[1,k] - u[1,k-1])^2 for k = 2:N)) +
        lambda_u[2] * ((u[2,1] - u_prev[2])^2 + sum((u[2,k] - u[2,k-1])^2 for k = 2:N)) +
        lambda_s_eta_min * s_eta_min_1^2 +
        lambda_s_eta_min * s_eta_min_2^2 +
        lambda_s_x * s_x1^2 +
        lambda_s_x * s_x2^2
    )

    return model

end