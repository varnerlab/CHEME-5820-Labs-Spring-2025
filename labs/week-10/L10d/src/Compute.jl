function _sample(model::MyRestrictedBoltzmannMachineModel, pass::MyFeedForwardPassModel, sₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)

    # initialize -
    W = model.W; # weight matrix
    b = model.b; # hidden bias vector
    a = model.a; # visible bias vector

    number_of_hidden_neurons = length(b); # number of hidden neurons
    S = zeros(Int, number_of_hidden_neurons, T);
    h = zeros(Float64, number_of_hidden_neurons); # input to the neurons
    IN = zeros(Float64, number_of_hidden_neurons); # input to the neurons
    is_ok_to_stop = false; # flag to stop the simulation

    # package initial state -
    v = copy(sₒ); # visible state is fixed, sample over the hidden state 
   
    # main loop -
    t = 1;
    while (is_ok_to_stop == false)

        # Compute the input to each neuron -
        for i ∈ 1:number_of_hidden_neurons
            IN[i] = dot(W[:, i], v) + b[i]; # compute the input for hidden node i -
        end

        # update the state of the neurons -
        for i ∈ 1:number_of_hidden_neurons
            pᵢ = (1 / (1 + exp(-β * IN[i])));  # compute the probability of flipping the i-th bit
            flag = Bernoulli(pᵢ) |> rand # flip the i-th bit with probability pᵢ
            h[i] = flag == 1 ? 1 : -1; # flip the i-th bit for the *next* state
        end
        
        S[:, t] .= copy(h); # store the current state in the S matrix
        
        if (t == T)
            is_ok_to_stop = true; # stop the simulation
        else
            t += 1; # increment the time step, and continue
        end
    end

    # return the results -    
    return S;
end

function _sample(model::MyRestrictedBoltzmannMachineModel, pass::MyFeedbackPassModel, sₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)


    # initialize -
    W = model.W; # weight matrix
    b = model.b; # hidden bias vector
    a = model.a; # visible bias vector

    number_of_visible_neurons = length(a); # number of visible neurons
    v = zeros(Float64, number_of_visible_neurons); # input to the neurons
    S = zeros(Int, number_of_visible_neurons , T);
    IN = zeros(Float64, number_of_visible_neurons ); # input to the neurons
    is_ok_to_stop = false; # flag to stop the simulation

    # package initial state -
    h = copy(sₒ); # *hidden* state is fixed, sample over the visible state 

    # main loop -
    t = 1;
    while (is_ok_to_stop == false)

        # Compute the input to each neuron -
        for i ∈ 1:number_of_visible_neurons
            IN[i] = dot(W[i,:], h) + a[i]; # compute the input for visible node i -
        end

        # update the state of the neurons -
        for i ∈ 1:number_of_visible_neurons
            pᵢ = (1 / (1 + exp(-β * IN[i])));  # compute the probability of flipping the i-th bit
            flag = Bernoulli(pᵢ) |> rand # flip the i-th bit with probability pᵢ
            v[i] = flag == 1 ? 1 : -1; # flip the i-th bit for the *next* state
        end
        S[:, t] .= copy(v); # store the current state in the S matrix
        
        if (t == T)
            is_ok_to_stop = true; # stop the simulation
        else
            t += 1; # increment the time step, and continue
        end
    end

    # return the results -    
    return S;
end


function simulate(model::MySimpleBoltzmannMachineModel, sₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)::Array{Int,2}
    
    # initialize storage -
    W = model.W; # weight matrix
    b = model.b; # bias vector

    number_of_neurons = length(sₒ);
    S = zeros(Int, number_of_neurons, T);
    is_ok_to_stop = false; # flag to stop the simulation
    h = zeros(Float64, number_of_neurons); # input to the neurons

    # package initial state -
    s = copy(sₒ); # initial state
    S[:, 1] .= s; # store the initial state in the S matrix

    # main loop -
    t = 2;
    while (is_ok_to_stop == false)

        # Compute the input to each neuron -
        for i ∈ 1:number_of_neurons
            h[i] = dot(W[i, :], s) + b[i]; # compute the input for node i -
        end

        # update the state of the neurons -
        for i ∈ 1:number_of_neurons
            pᵢ = (1 / (1 + exp(-2 * β * h[i])));  # compute the probability of flipping the i-th bit
            flag = Bernoulli(pᵢ) |> rand # flip the i-th bit with probability pᵢ
            s[i] = flag == 1 ? 1 : -1; # flip the i-th bit for the *next* state
        end
        
        S[:, t] .= copy(s); # store the current state in the S matrix
        
        if (t == T)
            is_ok_to_stop = true; # stop the simulation
        else
            t += 1; # increment the time step, and continue
        end
    end

    # return the results -    
    return S;
end

function simulate(model::MyRestrictedBoltzmannMachineModel, vₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)
    
    # forward pass - computes the hidden state given the visible state
    h = _sample(model, MyFeedForwardPassModel(), vₒ; T = T, β = β); # multiplex dispatch rocks!!
    v = _sample(model,  MyFeedbackPassModel(), h[:,end]; T = T, β = β); # multiplex dispatch rocks!!

    
    # return the results -
    return (v, h);
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

function energy(model::MySimpleBoltzmannMachineModel, s::Vector{Int})::Float64

    # initialize -
    W = model.W; # weight matrix
    b = model.b; # bias vector
    energy = -(1/2)*dot(s, W*s) - dot(b, s); # compute the energy of the state

    # return -
    return energy;
end

function energy(model::MyRestrictedBoltzmannMachineModel, v::Vector{Int}, h::Vector{Int})::Float64

    # initialize -
    W = model.W; # weight matrix
    b = model.b; # hidden bias vector
    a = model.a; # visible bias vector
    energy = dot(v, W*h) - dot(b, h) - dot(a, v); # compute the energy of the state

    # return -
    return energy;
end


function learn(model::MyRestrictedBoltzmannMachineModel, data::Array{Int64,2}, p::Categorical;
    maxnumberofiterations::Int = 100, T::Int = 100, β::Float64 = 1.0, batchsize::Int = 10, η::Float64 = 0.01,
    verbose::Bool = true)::MyRestrictedBoltzmannMachineModel

    # initialize -
    W = model.W # weight matrix
    b = model.b; # hidden bias vector
    a = model.a; # visible bias vector
    is_ok_to_stop = false; # flag to stop the simulation
    number_of_internal_steps = T; # number of internal steps that we take in the sampling step -
    counter = 1;

    # system size -
    number_of_visible_neurons = size(W, 1); # number of visible neurons
    number_of_hidden_neurons = size(W, 2); # number of hidden neurons

    # main loop -
    while (is_ok_to_stop == false)

        # generate some training data for this round -
        idx_batch_set = Set{Int64}();
        is_batch_set_full = false;
        while (is_batch_set_full == false)
        
            idx = rand(p); # generate a random index 
            push!(idx_batch_set, idx); # add to the set - this will fail if the index is already in the set
            if (length(idx_batch_set) == batchsize)
                is_batch_set_full = true; # ok to stop    
            end
        end
        idx_batch_vector = idx_batch_set |> collect |> sort;
        
        # process each of the batch elements -
        for i ∈ eachindex(idx_batch_vector)
            idx = idx_batch_vector[i]; # get the index
            xₒ = data[:, idx]; # get initial state that we will sample from
            
            # sample - 
            (v,h) = simulate(model, xₒ, T = number_of_internal_steps, β = β);

            # ok, so we have the visible and hidden states from sampling
            for j ∈ 1:number_of_visible_neurons
                for k ∈ 1:number_of_hidden_neurons
                    W[j,k] += η * (v[j, end] * h[k, end] - v[j, 1] * h[k, 1]); # update the weights - from GitHub Copilot - does this work?
                end
            end
        end


        if (verbose == true)
            println("Iteration: ", counter);
        end
        
        # check for convergence - should we stop?
        if (counter ≥ maxnumberofiterations)
            is_ok_to_stop = true; # stop the training 
        else
            counter += 1; # increment the counter
        end
    end

    # build a new model (that we'll return) after we estimate the W, a and b parameters -
    idmodel = build(MyRestrictedBoltzmannMachineModel, (
        W = W,
        b = b,
        a = a
    ));

    # return -
    return idmodel;
end


function learn(model::MySimpleBoltzmannMachineModel, data::Array{Int64,2}, sₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)::Tuple{Vector{Int}, Array{Int,2}, Array{Float64,1}}

    # fill me in -
    throw("Not implemented yet");

end

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