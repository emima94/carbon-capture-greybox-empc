
function get_name_to_idx_map(plant, w_Ta)

    if plant == "DTI"
        if w_Ta
            # DTI model including TlA_HX as state
            return Dict(
                :ca => (var = :x, idx = 1),
                :cd => (var = :x, idx = 2),
                :Tla_HX => (var = :x, idx = 3),
                :Tld_HX => (var = :x, idx = 4),
                :F => (var = :u, idx = 1),
                :Tla_HX_SP => (var = :u, idx = 2),
                :Tld_HX_SP => (var = :u, idx = 3),
                :cgina => (var = :d, idx = 1),
                :Fgina => (var = :d, idx = 2)
            )
        else
            # DTI model without TlA_HX as state
            return Dict(
                :ca => (var = :x, idx = 1),
                :cd => (var = :x, idx = 2),
                :Tld_HX => (var = :x, idx = 3),
                :F => (var = :u, idx = 1),
                :Tla_HX_SP => (var = :u, idx = 2),
                :Tld_HX_SP => (var = :u, idx = 3),
                :cgina => (var = :d, idx = 1),
                :Fgina => (var = :d, idx = 2)
            )
        end
    elseif plant == "Tiller"
        return Dict(
            :ca => (var = :x, idx = 1),
            :cd => (var = :x, idx = 2),
            :ct => (var = :x, idx = 3),
            :F => (var = :u, idx = 1),
            :Q => (var = :u, idx = 2),
            :cgina => (var = :d, idx = 1),
            :Fgina => (var = :d, idx = 2)
        )
    else
        error("Plant $(plant) not recognized.")
    end

end


function get_input_idxs(input_names, plant, w_Ta)
    
    input_idxs = (x = Int[], u = Int[], d = Int[])
    names_to_idxs_mapping = get_name_to_idx_map(plant, w_Ta)

    for name in input_names
        if haskey(names_to_idxs_mapping, name)
            var, idx = names_to_idxs_mapping[name]
            push!(input_idxs[var], idx)
        else
            error("Input name $(name) not found in mapping. Plant: $(plant), w_Ta: $(w_Ta).")
        end
    end

    return input_idxs
end


# function get_normalization_params(plant)

#     if plant == "DTI"
#     norm_params = Dict(
#         :ca => (mu=2.60, sigma=0.1),
#         :cd => (mu=2.10, sigma=0.1),
#         :cga => (mu=3.0, sigma=0.5),
#         :Tla_HX => (mu=37.5, sigma=5.0),
#         :Tld_HX => (mu=90.0, sigma = 10.0),
#         :F => (mu=0.5, sigma = 0.2),
#         :Tla_HX_SP => (mu=0.0, sigma=2.0),
#         :Tld_HX_SP => (mu=0.0, sigma = 2.0),
#         :Na => (mu = 300, sigma = 50),
#         :Nd => (mu=300, sigma=50),
#         :Q => (mu = 15.0, sigma = 5.0)
#         # :Na => (mu=125, sigma = 50),
#         # :Nd => (mu=125, sigma = 50),
#         # :Q => (mu = 15, sigma = 5.0)
#         # :Na => (mu=250, sigma = 50),
#         # :Nd => (mu=250, sigma = 50),
#         # :Q => (mu = 15, sigma = 5.0)
#     )
#     elseif plant == "Tiller"
#         norm_params = Dict(
#             :ca => (mu=2.60, sigma=0.1),
#             :cd => (mu=0.3, sigma=0.1),
#             :ct => (mu = 0.3, sigma = 0.1),
#             :cga => (mu=2.0, sigma=0.5),
#             :F => (mu=0.3, sigma = 0.1),
#             :Na => (mu=648, sigma=48),
#             :Nd => (mu=648, sigma=48),
#             :Q => (mu = 28.0, sigma = 2.0)
#         )

#     else
#         error("Unknown plant: $plant. Cannot set normalization parameters.")
#     end

#     return norm_params
# end


function get_normalization_params_high_mu(plant)

    if plant == "DTI"
    norm_params = Dict(
        :ca => (mu=2.60, sigma=0.1),
        :cd => (mu=2.10, sigma=0.1),
        :cga => (mu=3.0, sigma=0.5),
        :Tla_HX => (mu=37.5, sigma=5.0),
        :Tld_HX => (mu=90.0, sigma = 10.0),
        :F => (mu=0.5, sigma = 0.2),
        :Tla_HX_SP => (mu=0.0, sigma=2.0),
        :Tld_HX_SP => (mu=0.0, sigma = 2.0),
        :Na => (mu = 300, sigma = 50),
        :Nd => (mu=300, sigma=50),
        :Q => (mu = 15.0, sigma = 5.0)
        # :Na => (mu=125, sigma = 50),
        # :Nd => (mu=125, sigma = 50),
        # :Q => (mu = 15, sigma = 5.0)
        # :Na => (mu=250, sigma = 50),
        # :Nd => (mu=250, sigma = 50),
        # :Q => (mu = 15, sigma = 5.0)
    )
    elseif plant == "Tiller"
        norm_params = Dict(
            :ca => (mu=2.60, sigma=0.1),
            :cd => (mu=0.3, sigma=0.1),
            :ct => (mu = 0.3, sigma = 0.1),
            :cga => (mu=2.0, sigma=0.5),
            :F => (mu=0.3, sigma = 0.1),
            :Na => (mu=648, sigma=48),
            :Nd => (mu=648, sigma=48),
            :Q => (mu = 28.0, sigma = 2.0)
        )

    else
        error("Unknown plant: $plant. Cannot set normalization parameters.")
    end

    return norm_params
end

function get_normalization_params_mean_mu(plant)

    if plant == "DTI"
    norm_params = Dict(
        :ca => (mu=2.60, sigma=0.1),
        :cd => (mu=2.10, sigma=0.1),
        :cga => (mu=3.0, sigma=0.5),
        :Tla_HX => (mu=37.5, sigma=5.0),
        :Tld_HX => (mu=90.0, sigma = 10.0),
        :F => (mu=0.5, sigma = 0.2),
        :Tla_HX_SP => (mu=0.0, sigma=2.0),
        :Tld_HX_SP => (mu=0.0, sigma = 2.0),
        # :Na => (mu = 300, sigma = 50),
        # :Nd => (mu=300, sigma=50),
        # :Q => (mu = 15.0, sigma = 5.0)
        :Na => (mu=125, sigma = 50),
        :Nd => (mu=125, sigma = 50),
        :Q => (mu = 10, sigma = 5.0)
        # :Na => (mu=250, sigma = 50),
        # :Nd => (mu=250, sigma = 50),
        # :Q => (mu = 15, sigma = 5.0)
    )
    elseif plant == "Tiller"
        norm_params = Dict(
            :ca => (mu=2.60, sigma=0.1),
            :cd => (mu=0.3, sigma=0.1),
            :ct => (mu = 0.3, sigma = 0.1),
            :cga => (mu=2.0, sigma=0.5),
            :F => (mu=0.3, sigma = 0.1),
            :Na => (mu=648, sigma=48),
            :Nd => (mu=648, sigma=48),
            :Q => (mu = 28.0, sigma = 2.0)
        )

    else
        error("Unknown plant: $plant. Cannot set normalization parameters.")
    end

    return norm_params
end

#function get_ss_params(input_names, plant)
#    x_ss = Float64[]


function get_NN_norm_params(input_names, output_name, norm_params)

    mu_input = Float64[]
    sigma_input = Float64[]

    for names in input_names
        if haskey(norm_params, names)
            push!(mu_input, norm_params[names].mu)
            push!(sigma_input, norm_params[names].sigma)
        else
            error("Normalization parameters for $(names) not found.")
        end
    end

    mu_output = [norm_params[output_name].mu]
    sigma_output = [norm_params[output_name].sigma]

    return (input = (mu = mu_input, sigma = sigma_input), output = (mu = mu_output, sigma = sigma_output))
end

function get_NN_settings(NN_def, plant, w_Ta, norm_params)

    NN_settings = (;)

    NN_names = keys(NN_def)

    for NN_name in NN_names
        input_idxs = get_input_idxs(NN_def[NN_name].input_names, plant, w_Ta)
        input_func = NN_def[NN_name].input_func
        n_neurons = NN_def[NN_name].n_neurons
        norm_params_NN = get_NN_norm_params(NN_def[NN_name].input_names, NN_name, norm_params)

        NN_settings = (; NN_settings..., 
                    Symbol(NN_name) => (norm_params_NN..., n_neurons = n_neurons, input_func = input_func, input_idxs = input_idxs)
                )

    end

    return NN_settings

end

function get_obs_idx(plant, w_Ta, with_Q)

# Define indices of observations in the dataset used for training

    if plant == "DTI"
        if w_Ta
            obs_idx = [1,2,3]
        else
            obs_idx = [1,3]
        end
        if with_Q
            obs_idx = vcat(obs_idx, 4) # Q is the 4th observed variable for DTI
        end
    elseif plant == "Tiller"
        obs_idx = [1]
    else
        error("Plant $(plant) not recognized.")
    end

    return obs_idx
end

function initialize_parameters(datasets, NN_settings, p_fixed, plant, w_Ta; compute_ss_params = true)
    
    eval_Na_fun = p_fixed.eval_Na_fun
    norm_params = p_fixed.norm_params

    # Initialise Nd parameters
    NN_par_ss_init = initialize_NN_params_from_settings(NN_settings[:Nd])
    # Compute steady state parameters for Nd NN based on steady state data
    
    if compute_ss_params && !isnothing(datasets)
        Nd_output_var = [ds_i.misc[1, :] for ds_i in datasets]  # Get Na measurements for all datasets in the campaign
        #println("Computing steady state parameters for Nd NN...")
        Nd_params_ss = compute_init_par_ss(datasets, Nd_output_var, NN_par_ss_init, NN_settings[:Nd], plant, w_Ta, loss_ss, norm_params)
    else
        Nd_params_ss = NN_par_ss_init
    end
    
    p0 = (; Nd_params = Nd_params_ss)

    # If with_Q is true, also initialize Q NN parameters
    if haskey(NN_settings, :Q)
        NN_par_ss_init_Q = initialize_NN_params_from_settings(NN_settings[:Q])
        if compute_ss_params && !isnothing(datasets)             
            Q_output_var = [ds_i.Y[4, :] for ds_i in datasets]  # Get Q measurements for all datasets in the campaign
            #println("Computing steady state parameters for Q NN...")
            Q_params_ss = compute_init_par_ss(datasets, Q_output_var, NN_par_ss_init_Q, NN_settings[:Q], plant, w_Ta, loss_ss, norm_params)
        else
            Q_params_ss = NN_par_ss_init_Q
        end
        p0 = (; p0..., Q_params = Q_params_ss)
    end

    # Na parameters
    # Initialise Na parameters
    if eval_Na_fun == evaluate_Na_ca_Ta  # Only evaluate Na based on ca and if with_Ta: also Ta (k2 is add below)
         #p0 = (; p0..., k1 = 1.0, b = 1.0)
        Na_params_init = (k1 = 1.0, b = 0.0)
         if w_Ta
            #p0 = (; p0..., k2 = -0.01)
            Na_params_init = (; Na_params_init..., k2 = -0.01)
        end
    elseif eval_Na_fun == evaluate_Na_ca_cga_Ta #|| eval_Na_fun == evaluate_Na_ca_cga_Ta_002 # Evaluate Na based on ca, cg, and if with_Ta: also Ta (k3 is added below)
        #p0 = (; p0..., k1 = 1.0, k2 = 0.01, b = 1.0)
        Na_params_init = (k1 = -1.0, k2 = 1.0, b = 0.0)
        if w_Ta
            #p0 = (; p0..., k3 = -0.01)
            Na_params_init = (; Na_params_init..., k3 = -0.01)
        end
    elseif eval_Na_fun == evaluate_Na_ca_cd_cga_Ta
        Na_params_init = (k1 = -1.0, k2 = 1.0, k4 = -1.0, b = 0.0)
        if w_Ta
            Na_params_init = (; Na_params_init..., k3 = -0.01)
        end
    else
        error("Na evaluation function not recognized. Cannot initialize Na parameters.")
    end

    # Compute steady state parameters for Na NN based on steady state data
    
    if compute_ss_params && !isnothing(datasets)
        Na_output_var = [ds_i.misc[1, :] for ds_i in datasets]  # Get Na measurements for all datasets in the campaign
       # println("Computing steady state parameters for Na NN...")
        Na_params_ss = compute_init_par_ss(datasets, Na_output_var, Na_params_init, p_fixed, plant, w_Ta, loss_Na_ss, norm_params)
    else
        Na_params_ss = Na_params_init
    end
    # Add to p0
    p0 = (; p0..., Na_params = Na_params_ss)


    # Temperature rate parameters for DTI model
    if plant == "DTI"
        p0 = (; p0..., theta_D = 10.0)
        if w_Ta
            p0 = (; p0..., theta_A = 10.0)
        end
    end

    # Get initial state guess
    if plant == "DTI"
        if w_Ta
            x0_guess = [2.63; 2.10; 30; 90] # Initial guess for initial states, with TlA_HX as state, DTI plant
        else
            x0_guess = [2.63; 2.10; 90] # Initial guess for initial statesm, without TlA_HX as state, DTI plant
        end
    else
        x0_guess = [2.63; 0.3; 0.3] # Initial guess for initial states, Tiller plant
    end
    
    if isnothing(datasets)        
        x0_guess_vec = nothing
    else
        x0_guess_vec = [x0_guess for _ in eachindex(datasets)]
    end

    # Convert initial parameters and initial state guess to ComponentArrays for optimization

    θ0 = ComponentArray((p = p0, x0 = x0_guess_vec))

    return p0, x0_guess_vec, θ0

end 


function get_model_desc(camp_names, plant, with_Q, w_Ta, eval_Na_fun, NN_def, lambda_reg, norm_params_func)
    # Make model description based on config
    model_desc = "Model with plant=$(plant), with_Q=$(with_Q), w_Ta=$(w_Ta), eval_Na_fun=$(eval_Na_fun), NN_architectures=$(NN_def)"
    Nd_input_names = NN_def[:Nd][:input_names]
    Nd_n_neurons = NN_def[:Nd][:n_neurons]

    Nd_desc = "Nd($(join(string.(Nd_input_names), ", "))), n_neurons hidden layers: $(Nd_n_neurons)"
    Q_desc = if with_Q
        Q_input_names = NN_def[:Q][:input_names]
        Q_n_neurons = NN_def[:Q][:n_neurons]
        "Q($(join(string.(Q_input_names), ", "))), n_neurons hidden layers: $(Q_n_neurons)"
    else
        "No Q NN"
    end
    Na_desc = if eval_Na_fun == evaluate_Na_ca_cga_Ta
                    if w_Ta
                        "Na(ca, cga, Ta)"
                    else
                        "Na(ca, cga)"
                    end
                elseif eval_Na_fun == evaluate_Na_ca_Ta
                    if w_Ta
                        "Na(ca, Ta)"
                    else
                        "Na(ca)"
                    end
                elseif eval_Na_fun == evaluate_Na_ca_cd_cga_Ta
                    if w_Ta
                        "Na(ca, cd, cga, Ta)"
                    else
                        "Na(ca, cd, cga)"
                    end
                else
                    error("No description available for the specified Na evaluation function")
                end

    # Observation description
    obs_idx = get_obs_idx(plant, w_Ta, with_Q)
    obs_names_list = ["yga", "TlA_HX", "TlD_HX", "Q"] # Possible observed variables
    obs_names = obs_names_list[obs_idx]       

    #campaigns = unique([ds[i].camp for i in eachindex(ds)])
    model_type = if isempty(Nd_n_neurons) && (with_Q ? isempty(Q_n_neurons) : true)
        "Linear"
    else
        "Nonlinear"
    end


    # Model description:
    model_desc = "
    Plant: \t \t $(plant)
    Campaign:  \t \t $(join(string.(camp_names), ", "))" * 
#    Train idxs: \t $(join(string.(train_idxs), ", "))
    "
    Model type:  \t $(model_type)
    Na:  \t \t $(Na_desc)
    Nd:  \t \t $(Nd_desc)
    "
    if with_Q
        model_desc = model_desc * "Q:  \t \t $(Q_desc)
        "
    end
    model_desc = model_desc * "Obs:  \t \t $(join(obs_names, ", "))
    "
    model_desc = model_desc * "Lambda_reg: \t $(lambda_reg)
    Norm params: \t $(string(norm_params_func))
    "

    return model_desc

end

function get_train_idxs(train_idxs_list_in_sample, out_of_sample)

    train_idxs_list_out_of_sample = Vector{Vector{Int64}}[]
    for i in eachindex(train_idxs_list_in_sample)
        idx_i = [setdiff(train_idxs_list_in_sample[i], [j]) for j in train_idxs_list_in_sample[i]]
        push!(train_idxs_list_out_of_sample, idx_i)
    end

    train_idxs_list = [
        out_of_sample ?
        vcat([train_idxs_list_in_sample[i]], train_idxs_list_out_of_sample[i]) :
        [train_idxs_list_in_sample[i]]
        for i in eachindex(train_idxs_list_in_sample)
    ]

    return train_idxs_list
end

# Define NN input functions
function Nd_input_func(x)
    return x
end

function Q_input_func(x)
    return [x[1], x[2], x[3], x[4], x[3] - x[5]]
end

function Q_input_func_cd_Tld_F_TlD_SP(x)
    return [x[1], x[2], x[3], x[2] - x[4]]
end