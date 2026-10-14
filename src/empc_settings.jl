# ------------------------------------------------------------------------------------------
# Settings of the closed-loop EMPC tests at the DTI pilot plant (Section 7, Table 4).
# Same settings as get_closed_loop_settings_260415_01 (test #4) and
# get_closed_loop_settings_260416_01 (test #3) in src/settings/ of the original project.
# ------------------------------------------------------------------------------------------

"""
    empc_settings(scenario, time_start)

Settings of the closed-loop EMPC for the scenario "flexible" (test #4: minimize the energy
costs with a time-varying energy price) or "baseline" (test #3: minimize the energy use).
"""
function empc_settings(scenario, time_start)

    # Model: in-sample parameter estimate with lambda_reg = 0.1
    est = only(filter(e -> e.sample_type == "in-sample", load_parameter_estimates(LOG10_LAMBDA_CHOSEN)))
    p_opt = est.p
    x0_opt = est.x0
    model = build_dti_model(est.lambda_reg)
    ffun_bar, hfun, p_fixed = model.ffun_bar, model.hfun, model.p_fixed

    # Time horizon and control intervals
    t0 = 0.0
    tf = 6.0
    dt_control_interval = 10.0 / 60.0      # 10 min control interval [h]
    N = Int64(tf / dt_control_interval)

    # Discretization of the dynamics in the OCP (forward Euler, 30 s)
    Nx = N * 20 + 1
    dt_sim = (tf - t0) / (Nx - 1)

    # Weights in the objective function
    s_min, s_max = 0.0, 1e6
    lambda_F = 1e2              # Weight on changes in F
    lambda_Tld_HX_SP = 5        # Weight on changes in Td_sp
    lambda_s_x = 1e6            # Weight on slack variables for state constraints
    lambda_s_eta_min = 1e6      # Weight on slack variable for the capture efficiency constraint
    if scenario == "flexible"
        lambda = ([lambda_F, lambda_Tld_HX_SP], lambda_s_eta_min, lambda_s_x)
        OCP_model_func = generate_JuMP_model_EMPC_DTI_closed_loop       # Minimize energy costs
    elseif scenario == "baseline"
        lambda_Q = 80.0         # Weight on the reboiler duty Q
        lambda = ([lambda_F, lambda_Tld_HX_SP], lambda_Q, lambda_s_eta_min, lambda_s_x)
        OCP_model_func = generate_JuMP_model_ref_MPC_DTI_closed_loop    # Minimize energy use
    else
        error("Unknown scenario: $(scenario)")
    end

    # Input bounds and bounds on input changes per hour
    U_min = [0.2, 85]
    U_max = [0.8, 95]
    lb = (U_min, s_min, -s_max)
    ub = (U_max, s_max, s_max)
    du = ([-0.6, -12], [0.6, 12])

    # Disturbances (cgin [mol/m3], Fin [m3/h]) used for the initial steady state
    dist_base = [5.9, 80.0]

    # Energy price for two horizons [EUR/MWh]
    p_el_horizon1 = 80 .- 60 .* sin.(2π .* (0:(N-1)) * dt_control_interval / tf)
    p_el = vcat(p_el_horizon1, p_el_horizon1)

    # Minimum capture efficiency [%]
    eta_min = 30.0

    # EKF settings
    nx = length(x0_opt[1])
    ny = length(p_fixed.hfun(x0_opt[1], [0.5, NaN, 90], [0; 0], p_opt, p_fixed))
    sx = 0.1 * ones(nx)
    Rv = [3.0, 1.0, 10.0].^2 .* I(ny)
    dt_sim_euler = 10 / 3600
    p_fixed_ekf = (; p_fixed..., nx = nx, Rv = Rv, sx = sx, dt = dt_sim_euler)

    # Model functions with the two inputs u = [F, Td_sp] (the absorber temperature is not used)
    ffun_2du = (x, u, d, p, p_fixed) -> ffun_bar(x, [u[1]; NaN; u[2]], d, p, p_fixed)
    hfun_2du = (x, u, d, p, p_fixed) -> hfun(x, [u[1]; NaN; u[2]], d, p, p_fixed)
    ffun_ekf = (x, u, d, p, p_fixed) -> moment_equations_EKF(x, u, d, p, p_fixed, ffun_2du)
    hfun_ekf = (x, u, d, p, p_fixed) -> hfun_2du(x, u, d, p, p_fixed)[[1, 3, 4]]
    ffun_JuMP = (x, u, d, p, p_fixed) -> ffun_with_accum_CO2(x, u, d, p, p_fixed, ffun_2du, hfun_2du)
    hfun_Q_JuMP = (x, u, d, p, p_fixed) -> hfun_2du(x[1:3], u, d, p, p_fixed)[4]
    hfun_cap_eff_JuMP = (x, u, d, p, p_fixed) -> hfun_2du(x, u, d, p, p_fixed)[7]

    # Initial inputs: optimal steady state with eta = eta_min
    p_scenario_ss = (
        cap_eff_ref = eta_min,
        x_ss_guess = x0_opt[1],
        u_ss_guess = [0.5, 90],
        lb = U_min,
        ub = U_max,
        ffun = ffun_2du,
        hfun_cap_eff = (x, u, d, p, p_fixed) -> hfun_2du(x, u, d, p, p_fixed)[7],
        hfun_Q = (x, u, d, p, p_fixed) -> hfun_2du(x, u, d, p, p_fixed)[4],
        dist = dist_base,
        lambda = (1, 10)
    )
    model_ss = generate_JuMP_model_ref_SS(p_scenario_ss, p_opt, p_fixed)
    optimize!(model_ss)
    u_prev = collect(value.(model_ss[:u_ss]))

    funcs = (ffun_2du = ffun_2du, hfun_2du = hfun_2du, ffun_ekf = ffun_ekf, hfun_ekf = hfun_ekf,
             ffun_JuMP = ffun_JuMP, hfun_Q_JuMP = hfun_Q_JuMP, hfun_cap_eff_JuMP = hfun_cap_eff_JuMP,
             OCP_model_func = OCP_model_func)

    return (funcs = funcs, p_opt = p_opt, p_fixed = p_fixed, x0_opt = x0_opt, time_start = time_start,
            dt_control_interval = dt_control_interval, N = N, Nx = Nx, dt_sim = dt_sim, lambda = lambda,
            lb = lb, ub = ub, du = du, p_el = p_el, eta_min = eta_min, p_fixed_ekf = p_fixed_ekf,
            u_prev = u_prev, dist_base = dist_base, nx = nx)
end
