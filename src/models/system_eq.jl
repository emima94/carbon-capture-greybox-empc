
function ffun_bar_general(x, u, d, p, p_fixed)

    # Extract model settings from p_fixed
    w_Ta = p_fixed.w_Ta
    plant = p_fixed.plant
    eval_Na_fun = p_fixed.eval_Na_fun
    

    if plant == "DTI"
        if w_Ta # TlA_HX is included as a state
            ca, cd, Tla_HX, Tld_HX = x[1:4]
        else
            ca, cd, Tld_HX = x[1:3]
        end
        Va = 0.25
        Vd = 0.25
    else # plant == "Tiller"
        ca, cd, ct = x[1:3]
        Va = 0.17
        Vd = 0.17
        Vt = 0.17
    end

    # Adaptive model?
    if haskey(p_fixed, :adaptive_model)
        adaptive_model = p_fixed.adaptive_model
    else
        adaptive_model = false
    end
    if adaptive_model
        n_d_adaptive = p_fixed.n_d_adaptive
        idx_d_adaptive = length(x) - n_d_adaptive + 1 : length(x) # Indices of adaptive disturbance in state vector
        d_adaptive = x[idx_d_adaptive] # Last element of state vector is the adaptive parameter
    end
    

    mol2kmol = 1e-3

    # Inputs #
    if plant == "DTI"
        F, Tla_HX_SP, Tld_HX_SP = u
    else # plant == "Tiller"
        F, Q = u
    end

    # Disturbances #
    cgina, Fgina = d

    # Evaluate Na NN
    #Na = eval_Na_fun(p, p_fixed, x, u, d)[1]
    Na = eval_Na_fun(p.Na_params, p_fixed, x, u, d)[1] # Add adaptive term to Na

    # Evaluate Nd
    Nd = evaluate_NN_outer(p.Nd_params, p_fixed.Nd, x, u, d)

    #if adaptive_model
        #scale_d_adaptive = Vd / mol2kmol
        #Nd += d_adaptive[1] * scale_d_adaptive # Add adaptive term to Na
        #Nd += d_adaptive * scale_d_adaptive # Add adaptive term to Nd
    #end

    # ODEs:
    if plant == "DTI"
        if w_Ta
            dx = [(F * (cd - ca) + Na * mol2kmol) / Va,
                (F * (ca - cd) - Nd * mol2kmol) / Vd,
                p.theta_A * (Tla_HX_SP - Tla_HX), # Simple first-order dynamics for Tla_HX
                p.theta_D * (Tld_HX_SP - Tld_HX) # Simple first-order dynamics for Tld_HX
            ] 
        else
            dx = [(F * (cd - ca) + Na * mol2kmol) / Va,
                (F * (ca - cd) - Nd * mol2kmol) / Vd,
                p.theta_D * (Tld_HX_SP - Tld_HX) # Simple first-order dynamics for Tld_HX
            ] 
        end
    else # plant == "Tiller"
        dx = [(F * (ct - ca) + Na * mol2kmol) / Va,
            (F * (ca - cd) - Nd * mol2kmol) / Vd,
            F * (cd - ct) / Vt
            ] 
    end

    if adaptive_model
        dx[1] += d_adaptive[1]
        if n_d_adaptive > 1
             dx[2] += d_adaptive[2]
        end
        dx = vcat(dx, zeros(n_d_adaptive)) # Add zero dynamics for the adaptive parameter
    end

    return dx

end

function hfun_general(x, u, d, p, p_fixed)


    # Extract model settings from p_fixed
    plant = p_fixed.plant
    w_Ta = p_fixed.w_Ta   
    with_Q = p_fixed.with_Q
    eval_Na_fun = p_fixed.eval_Na_fun

    if plant == "DTI"
        if w_Ta # Tla_HX is included as a state
            ca, cd, Tla_HX, Tld_HX = x[1:4]
        else
            ca, cd, Tld_HX = x[1:3]
            Tla_HX = NaN # Placeholder value when Tla_HX is not a state
        end
        ct = NaN # Not applicable for DTI
        Va = 0.25
        Vd = 0.25
    else # plant == "Tiller"
        ca, cd, ct = x[1:3]
        Tla_HX = NaN # Not applicable for Tiller
        Tld_HX = NaN # Not applicable for Tiller
        Va = 0.17
        Vd = 0.17
    end

    # Adaptive model?
    if haskey(p_fixed, :adaptive_model)
        adaptive_model = p_fixed.adaptive_model
    else
        adaptive_model = false
    end
    if adaptive_model
        n_d_adaptive = p_fixed.n_d_adaptive
        idx_d_adaptive = length(x) - n_d_adaptive + 1 : length(x) # Indices of adaptive disturbance in state vector
        d_adaptive = x[idx_d_adaptive] # Last element of state vector is the adaptive parameter
    else
        d_adaptive = 0.0
    end

   # Conversion Constants
    R = 8.314 # J/(mol*K)
    Ta = 313.15 # K
    Pa = 101325 # Pa
    M_CO2 = 44.01 # g/mol
    g2kg = 1e-3 # g to kg
    mol2kmol = 1e-3

    # Inputs #
    if plant == "DTI"
        F, TlA_HX_SP, TlD_HX_SP = u
    else # plant == "Tiller"
        F, Q = u
    end

    # Disturbances #
    cgina, Fgina = d

    # Evaluate Na
    #Na = eval_Na_fun(p, p_fixed, x, u, d)[1]
    Na = eval_Na_fun(p.Na_params, p_fixed, x, u, d)[1]
    
    # Evaluate Nd
    Nd = evaluate_NN_outer(p.Nd_params, p_fixed.Nd, x, u, d) 

    #if adaptive_model
        #scale_d_adaptive_1 = Vd / mol2kmol
        #Nd += d_adaptive[1] * scale_d_adaptive_1 # Add adaptive term to Nd
        #Nd += d_adaptive[1] * scale_d_adaptive_1 # Add adaptive term to Nd
    #end

    # Mass balance, CO2, gas phase, absorber, steady state:
    cga = cgina - Na / Fgina
    
    # Total mol concentration in gas phase (assuming ideal gas behavior and constant T and P)
    cga_tot = Pa / (R * Ta)

    # Inlet CO2 concentration [vol%]:
    ygina = cgina / cga_tot * 100

    # Output CO2 concentration [vol%]:
    yga = cga / cga_tot * 100

    if adaptive_model
        scale_d_adaptive_1 = 10.0
        scale_d_adaptive_1_mult = 1.0
        #yga += d_adaptive[1] * scale_d_adaptive_1 # Add adaptive term to output CO2 concentration (converted to vol%)
        #yga = yga * ((d_adaptive[1]+1) * scale_d_adaptive_1_mult) + d_adaptive[2] * scale_d_adaptive_1_add # Scale output CO2 concentration by adaptive term (since yga is on a different scale than Na, we multiply instead of add)
    end

    # Capture efficiency:
    cap_eff = (ygina - yga) / ygina * 100

    # Compute adaptive Na for monitoring purposes (not used in ODEs, just for monitoring how the adaptive term is affecting Na)
    #cga_adaptive = yga / 100 * cga_tot # Convert output CO2 concentration back to mol/m^3 for use in adaptive term
    #Na_adaptive = Fgina * (cgina - cga_adaptive) # Calculate Na based on adaptive output concentration for monitoring
    #Na = Na_adaptive # Override Na with adaptive version for use in ODEs

    # Capture rate [kg/h]
    ma = Na * M_CO2 * g2kg

    # Reboiler duty, Q [kW]
    if plant == "DTI"
        if with_Q
            Q = evaluate_NN_outer(p.Q_params, p_fixed.Q, x, u, d)
            if adaptive_model
                scale_d_adaptive_2 = 10.0
                #scale_d_adaptive_2_mult = 1.0
                #scale_d_adaptive_2_add = 10.0
                #Q += d_adaptive[2] * scale_d_adaptive_2 # Add second adaptive term to Q if using adaptive model.
                #Q = Q * ((d_adaptive[3]+1) * scale_d_adaptive_2_mult) + d_adaptive[4] * scale_d_adaptive_2_add # Scale Q by adaptive term if using adaptive model (since Q is on a different scale than yga, we multiply instead of add)
            end
        else 
            Q = 0.0
        end
    # else # plant == "Tiller"
        #Q = Q
    end

    # Specific Reboiler Duty [MJ/kg CO2]
    kJ2MJ = 1e-3
    s2h = 3600
    SRD = (Q * kJ2MJ * s2h) / ma

    #z = [yga, Tla_HX, Tld_HX, Q, Na, Nd, cap_eff, SRD, ygina, ma, ca, cd, ct]
    z = [yga, Tla_HX, Tld_HX, Q, Na, Nd, cap_eff, SRD, ygina, ma, ca, cd, ct, d_adaptive] # Include adaptive parameter in output for monitoring

    return z

end


