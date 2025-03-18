
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
        s[j] = sign(transpose(W[j,:])*s - b[j]); # update the state

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
    decode(simulationstate::Array{T,1}; number_of_rows::Int64 = 28, number_of_cols::Int64 = 28) -> Array{T,2}
"""
function decode(simulationstate::Array{T,1}; 
    number_of_rows::Int64 = 28, number_of_cols::Int64 = 28)::Array{T,2} where T <: Number
    
    # initialize -
    reconstructed_image = Array{Int32,2}(undef, number_of_rows, number_of_cols);
    linearindex = 1;
    for row ∈ 1:number_of_rows
        for col ∈ 1:number_of_cols
            s = simulationstate[linearindex];
            if (s == -1)
                reconstructed_image[row,col] = 0;
            else
                reconstructed_image[row,col] = 1;
            end
            linearindex+=1;
        end
    end
    
    # return 
    return reconstructed_image
end