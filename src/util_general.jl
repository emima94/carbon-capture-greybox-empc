
function simulate_system(ffun, x0, u, dist, p, p_fixed, t_vec)
    
    # Allocate memory for state trajectory
    nx = length(x0)
    x = zeros(nx, length(t_vec))
    x[:,1] = x0
    
    for i in 1:(length(t_vec)-1)
        dt = t_vec[i+1] - t_vec[i]
        f = ffun(x[:,i], u[:,i], dist[:,i], p, p_fixed)
        x[:,i+1] = x[:,i] + dt * f
    end

    return x
end

