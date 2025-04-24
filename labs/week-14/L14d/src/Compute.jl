
"""
    energy(W::Array{T,2}, α::Array{T,1}, s::Array{T,1}) -> T where T <: Number
"""
function _energy(s::Array{<: Number,1}, W::Array{<:Number,2}, b::Array{<:Number,1})::Float32
    
    # initialize -
    tmp_energy_state = 0.0;
    number_of_states = length(s);

    # main loop -
    tmp = transpose(b)*s; # alias for the bias term
    for i ∈ 1:number_of_states
        for j ∈ 1:number_of_states
            tmp_energy_state += W[i,j]*s[i]*s[j];
        end
    end
    energy_state = -(1/2)*tmp_energy_state + tmp;

    # return -
    return energy_state;
end

"""
    softmax(x::Array{T,1}) -> Array{T,1} where T <: Number

Compute the softmax of a vector `x` and returns a probability vector. We subtract the maximum value from the vector to avoid overflow.

### Arguments
- `x::Array{T,1}`: a vector of length `n`.

### Returns
- `θ::Array{T,1}`: a probability vector of length `n`.

"""
function softmax(x::Array{T,1})::Array{T,1} where T <: Number
    
    # compute the softmax of a vector
    number_of_elements = length(x);
    θ = zeros(T, number_of_elements);
    m = maximum(x); # max of the vector

    x̂ = x .- m; # subtract the maximum value

    # compute the softmax -
    for i ∈ 1:number_of_elements
        θ[i] = exp(x̂[i]);
    end

    θ = θ/sum(θ);

    # return -
    return θ;
end

"""
    ⊗(a::Array{Float64,1},b::Array{Float64,1}) -> Array{Float64,2}

Compute the outer product of two vectors `a` and `b` and returns a matrix.

### Arguments
- `a::Array{Float64,1}`: a vector of length `m`.
- `b::Array{Float64,1}`: a vector of length `n`.

### Returns
- `Y::Array{Float64,2}`: a matrix of size `m x n` such that `Y[i,j] = a[i]*b[j]`.
"""
function ⊗(a::Array{T,1}, b::Array{T,1})::Array{T,2} where T <: Number

    # initialize -
    m = length(a)
    n = length(b)
    Y = zeros(m,n)

    # main loop 
    for i ∈ 1:m
        for j ∈ 1:n
            Y[i,j] = a[i]*b[j]
        end
    end

    # return 
    return Y
end


"""
    recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{T,1}; 
        maxiterations::Int = 1000, trueindex::Int = 2) -> Tuple{Dict{Int64, Array{Int32,1}}, Dict{Int64, Float32}} where T <: Number

This method takes a Hopfield network model and a random state and returns a dictionary of frames and a dictionary of energies what hold data for each iteration.

### Arguments
- `model::MyClassicalHopfieldNetworkModel`: a Hopfield network model.
- `sₒ::Array{T,1}`: a random state.
- `maxiterations::Int`: the maximum number of iterations.
- `trueindex::Int`: the index of the true state (we index into the energy dictionary on model, use to stop the simulation early).

### Returns
A tuple of two dictionaries:
- `frames::Dict{Int64, Array{Int32,1}}`: a dictionary of frames.
- `energydictionary::Dict{Int64, Float32}`: a dictionary of energies.
"""
function recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{T,1}; 
    maxiterations::Int = 1000, trueindex::Int = 2)::Tuple{Dict{Int64, Array{Int32,1}}, Dict{Int64, Float32}} where T <: Number

    # initialize -
    W = model.W;
    b = model.b;
    true_energy = model.energy[trueindex];

    # initialize -
    frames = Dict{Int64, Array{Int32,1}}();
    energydictionary = Dict{Int64, Float32}();
    has_converged = false;

    # setup -
    frames[0] = copy(sₒ); # copy the initial random state
    energydictionary[0] = _energy(sₒ,W, b); # initial energy
    s = copy(sₒ); # initial state
    iteration_counter = 1;
    while (has_converged == false)
        
        j = rand(1:number_of_pixels); # select a random pixel
        w = W[j,:]; # get the weights
        s[j] = sign(dot(w,s) - b[j]); # update the state

        energydictionary[iteration_counter] = _energy(s, W, b);
        frames[iteration_counter] = copy(s); # save a copy
        
            
        if ((energydictionary[iteration_counter] ≈ true_energy) || (iteration_counter ≥  maxiterations))
            has_converged = true;
        end
        iteration_counter += 1;
    end
            
    # return 
    frames, energydictionary
end


"""
    recover(model::MyModernHopfieldNetworkModel, sₒ::Array{T,1}; 
        maxiterations::Int64 = 1000, ϵ::Float64 = 1e-10) where T <: Number

This method takes a moderm Hopfield network model and a random state 
and state `s`, the frames and the probabilities of the states at each iteration.

### Arguments
- `model::MyModernHopfieldNetworkModel`: a Hopfield network model.
- `sₒ::Array{T,1}`: a random state.
- `maxiterations::Int64`: the maximum number of iterations.
- `ϵ::Float64`: the convergence threshold.

### Returns
- `s::Array{T,1}`: the final state.
- `frames::Dict{Int64, Array{Float32,1}}`: a dictionary of frames (states at each iteration).
- `probability::Dict{Int64, Array{Float64,1}}`: a dictionary of probabilities (probabilities of the states at each iteration).
"""
function recover(model::MyModernHopfieldNetworkModel, sₒ::Array{T,1}; 
    maxiterations::Int64 = 1000, ϵ::Float64 = 1e-10) where T <: Number

    # initialize -
    X = model.X; # data matrix from the model. This holds the memories on the columns
    β = model.β; # beta parameter (inverse temperature)

    frames = Dict{Int64, Array{Float32,1}}(); # save the iterations -
    probability = Dict{Int64, Array{Float64,1}}(); # save the probabilities
    frames[0] = copy(sₒ); # copy the initial random state
    probability[0] = softmax(β*transpose(X)*sₒ); # initial probability
    should_stop_iteration = false; # flag to stop the iteration
    iteration_counter = 1; # iteration counter

    # loop -
    s = copy(sₒ); # initial state
    Δ = Inf; # initial delta
    while (should_stop_iteration == false)
        
        p = softmax(β*transpose(X)*s); # compute the probabilities
        s = X*p; # update the state
        
        frames[iteration_counter] = copy(s); # save a copy of the state in the frames dictionary
        probability[iteration_counter] = p; # save the probabilities in the probability dictionary

        # first: compute the difference between the current and previous probabilities
        if (iteration_counter > 1)
            Δ = norm(probability[iteration_counter] - probability[iteration_counter-1]);
        end

        # next: check for convergence. If we are out of iterations or the difference is small, we stop
        if (iteration_counter >= maxiterations || Δ ≤ ϵ)
            should_stop_iteration = true;
        else
            iteration_counter += 1; # increment the iteration counter, we are not done yet. Keep going.
        end
    end

    # return -
    return s,frames,probability
end


"""
    decode(simulationstate::Array{T,1}; number_of_rows::Int64 = 28, number_of_cols::Int64 = 28) -> Array{T,2}

This function takes a simulation state vector and returns a 2D array of the same type.

### Arguments
- `simulationstate::Array{T,1}`: a simulation state vector.
- `number_of_rows::Int64`: the number of rows in the output array.
- `number_of_cols::Int64`: the number of columns in the output array.

### Returns
- `reconstructed_image::Array{T,2}`: a 2D array of the same type as the input vector.
"""
function decode(simulationstate::Array{T,1}; 
    number_of_rows::Int64 = 28, number_of_cols::Int64 = 28)::Array{T,2} where T <: Number
    
    # initialize -
    reconstructed_image = Array{T,2}(undef, number_of_rows, number_of_cols);
    linearindex = 1;
    for row ∈ 1:number_of_rows
        for col ∈ 1:number_of_cols
            reconstructed_image[row,col] = simulationstate[linearindex];
            linearindex+=1;
        end
    end
    
    # return 
    return reconstructed_image
end