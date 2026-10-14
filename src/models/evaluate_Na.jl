
# function evaluate_Na_ca_cga_Ta_old(p, p_fixed, x, u, d)

#     " 
#     Equations overview:

#     Steady state mass balance for cga:
#         0 = Fgina * (cgina - cga) - Na
    
#     Capture rate, absorption, Na_norm:
#         Na_norm = k1 * (ca_star_norm - ca_norm)
    
#     Equilibrium concentration, ca:     
#         ca_star_norm = b + k2 * cga_norm + k3 * Ta_norm
    
#     "
    
#     # Does state include Tla?
#     w_Ta = p_fixed.w_Ta;

#     # Get normalization parameters
#     norm_params = p_fixed.norm_params
#     mu_ca, sigma_ca = norm_params[:ca]
#     mu_cga, sigma_cga = norm_params[:cga]
#     mu_Na, sigma_Na = norm_params[:Na]

#     # Extract states
#     ca = x[1]

#     cgina = d[1]
#     Fgina = d[2]

#     # Normalize inputs
#     ca_norm = (ca - mu_ca) / sigma_ca

#     # Extract parameters
#     k1 = p.k1
#     k2 = p.k2
#     b = p.b

#     # If Tla is a state:
#     if w_Ta
#         mu_Ta, sigma_Ta = norm_params[:Tla_HX]
#         Ta = x[3]
#         Ta_norm = (Ta - mu_Ta) / sigma_Ta
#         k3 = p.k3
#     else
#         Ta_norm = 0.0
#         k3 = 0.0
#     end

#     b_bar = k1 * b
#     k1_bar = -k1
#     k2_bar = k1 * k2
#     k3_bar = k1 * k3

#     alpha = sigma_Na * k2_bar / sigma_cga

#     cga = (Fgina * cgina - sigma_Na * (b_bar + k1_bar * ca_norm + k3_bar * Ta_norm) - mu_Na + alpha * mu_cga) / (Fgina + alpha)

#     cga_norm = (cga - mu_cga) / sigma_cga

#     ca_star_norm = b + k2 * cga_norm + k3 * Ta_norm

#     Na_norm = k1 * (ca_star_norm - ca_norm)

#     Na = Na_norm * sigma_Na + mu_Na

#     return Na, cga

# end



function evaluate_Na_ca_Ta(p, p_fixed, x, u, d)

    " 
    Equations overview:

    Steady state mass balance for cga:
        0 = Fgina * (cgina - cga) - Na
    
    Capture rate, absorption, Na_norm:
        Na_norm = k1 * (b - ca_norm)
    
    "
    
    # Does state include Tla?
    w_Ta = p_fixed.w_Ta;

    # Get normalization parameters
    norm_params = p_fixed.norm_params
    mu_ca, sigma_ca = norm_params[:ca]
    mu_Na, sigma_Na = norm_params[:Na]

    # Extract states
    ca = x[1]

    cgina = d[1]
    Fgina = d[2]

    # Normalize inputs
    ca_norm = (ca - mu_ca) / sigma_ca

    # Extract parameters
    k1 = p.k1
    b = p.b

    # If Tla is a state:
    if w_Ta
        mu_Ta, sigma_Ta = norm_params[:Tla_HX]
        Ta = x[3]
        Ta_norm = (Ta - mu_Ta) / sigma_Ta
        k2 = p.k2
    else
        Ta_norm = 0.0
        k2 = 0.0
    end

#    Na_norm = k1 * (b - ca_norm)
    Na_norm = k1 * ((b + k2 * Ta_norm) - ca_norm)

    Na = Na_norm * sigma_Na + mu_Na

    cga = (cgina * Fgina - Na) / Fgina

    return Na, cga

end


function evaluate_Na_ca_cga_Ta(p, p_fixed, x, u, d)

    " 
    Equations overview:

    Steady state mass balance for cga:
        0 = Fgina * (cgina - cga) - Na
    
    Capture rate, absorption, Na_norm:
        Na_norm = b + k1 * ca_norm + k2 * cga_norm + k3 * Ta_norm
    
    "
    
    # Does state include Tla?
    w_Ta = p_fixed.w_Ta;

    # Get normalization parameters
    norm_params = p_fixed.norm_params
    mu_ca, sigma_ca = norm_params[:ca]
    mu_cga, sigma_cga = norm_params[:cga]
    mu_Na, sigma_Na = norm_params[:Na]

    # Extract states
    ca = x[1]

    cgina = d[1]
    Fgina = d[2]

    # Normalize inputs
    ca_norm = (ca - mu_ca) / sigma_ca

    # Extract parameters
    k1 = p.k1
    k2 = p.k2
    b = p.b

    # If Tla is a state:
    if w_Ta
        mu_Ta, sigma_Ta = norm_params[:Tla_HX]
        Ta = x[3]
        Ta_norm = (Ta - mu_Ta) / sigma_Ta
        k3 = p.k3
    else
        Ta_norm = 0.0
        k3 = 0.0
    end


    alpha = sigma_Na * k2 / sigma_cga

    cga = (Fgina * cgina - sigma_Na * (b + k1 * ca_norm + k3 * Ta_norm) - mu_Na + alpha * mu_cga) / (Fgina + alpha)
    cga_norm = (cga - mu_cga) / sigma_cga

    Na_norm = b + k1 * ca_norm + k2 * cga_norm + k3 * Ta_norm

    Na = Na_norm * sigma_Na + mu_Na

    return Na, cga

end


function evaluate_Na_ca_cd_cga_Ta(p, p_fixed, x, u, d)

    " 
    Equations overview:

    Steady state mass balance for cga:
        0 = Fgina * (cgina - cga) - Na
    
    Capture rate, absorption, Na_norm:
        Na_norm = b + k1 * ca_norm + k2 * cga_norm + k3 * Ta_norm
    
    "
    
    # Does state include Tla?
    w_Ta = p_fixed.w_Ta;

    # Get normalization parameters
    norm_params = p_fixed.norm_params
    mu_ca, sigma_ca = norm_params[:ca]
    mu_cd, sigma_cd = norm_params[:cd]
    mu_cga, sigma_cga = norm_params[:cga]
    mu_Na, sigma_Na = norm_params[:Na]

    # Extract states
    ca = x[1]
    cd = x[2]

    cgina = d[1]
    Fgina = d[2]

    # Normalize inputs
    ca_norm = (ca - mu_ca) / sigma_ca
    cd_norm = (cd - mu_cd) / sigma_cd

    # Extract parameters
    k1 = p.k1
    k2 = p.k2
    k4 = p.k4
    b = p.b

    # If Tla is a state:
    if w_Ta
        mu_Ta, sigma_Ta = norm_params[:Tla_HX]
        Ta = x[3]
        Ta_norm = (Ta - mu_Ta) / sigma_Ta
        k3 = p.k3
    else
        Ta_norm = 0.0
        k3 = 0.0
    end


    alpha = sigma_Na * k2 / sigma_cga

    cga = (Fgina * cgina - sigma_Na * (b + k1 * ca_norm + k3 * Ta_norm + k4 * cd_norm) - mu_Na + alpha * mu_cga) / (Fgina + alpha)
    cga_norm = (cga - mu_cga) / sigma_cga

    Na_norm = b + k1 * ca_norm + k2 * cga_norm + k3 * Ta_norm + k4 * cd_norm

    Na = Na_norm * sigma_Na + mu_Na

    return Na, cga

end

# function evaluate_Na_ca_cga_Ta_003(p, p_fixed, x, u, d)

#     " 
#     Equations overview:

#     Steady state mass balance for cga:
#         0 = Fgina * (cgina - cga) - Na
    
#     Capture rate, absorption, Na_norm:
#         Na_norm = b + k1 * ca_norm + k2 * cga_norm + k3 * Ta_norm
    

#     With new strucutre of parameters.

#     "
    
#     # Does state include Tla?
#     w_Ta = p_fixed.w_Ta;

#     # Get normalization parameters
#     norm_params = p_fixed.norm_params
#     mu_ca, sigma_ca = norm_params[:ca]
#     mu_cga, sigma_cga = norm_params[:cga]
#     mu_Na, sigma_Na = norm_params[:Na]

#     # Extract states
#     ca = x[1]

#     cgina = d[1]
#     Fgina = d[2]

#     # Normalize inputs
#     ca_norm = (ca - mu_ca) / sigma_ca

#     # Extract parameters
#     k1 = p.k1
#     k2 = p.k2
#     b = p.b

#     # If Tla is a state:
#     if w_Ta
#         mu_Ta, sigma_Ta = norm_params[:Tla_HX]
#         Ta = x[3]
#         Ta_norm = (Ta - mu_Ta) / sigma_Ta
#         k3 = p.k3
#     else
#         Ta_norm = 0.0
#         k3 = 0.0
#     end


#     alpha = sigma_Na * k2 / sigma_cga

#     cga = (Fgina * cgina - sigma_Na * (b + k1 * ca_norm + k3 * Ta_norm) - mu_Na + alpha * mu_cga) / (Fgina + alpha)
#     cga_norm = (cga - mu_cga) / sigma_cga

#     Na_norm = b + k1 * ca_norm + k2 * cga_norm + k3 * Ta_norm

#     Na = Na_norm * sigma_Na + mu_Na

#     return Na, cga

# end