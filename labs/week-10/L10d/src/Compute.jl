function _sample(model::MyRestrictedBoltzmannMachineModel, pass::FeedForwardPassModel, sₒ::Vector{Int}; 
    T::Int = 100, β::Float = 1.0)

    # initialize -
    W = model.W; # weight matrix
    b = model.b; # hidden bias vector
    a = model.a; # visible bias vector

    number_of_hidden_neurons = length(b); # number of hidden neurons
    S = zeros(Int, number_of_hidden_neurons, T);
    IN = zeros(Float64, number_of_hidden_neurons); # input to the neurons
    is_ok_to_stop = false; # flag to stop the simulation

    # package initial state -
    v = copy(sₒ); # visible state is fixed, sample over the hidden state 
    S[:, 1] .= v; # store the initial state in the S matrix

    # main loop -
    t = 2;
    while (is_ok_to_stop == false)

        # Compute the input to each neuron -
        for i ∈ 1:number_of_hidden_neurons
            IN[i] = dot(W[:, i], v) + b[i]; # compute the input for hidden node i -
        end

        # update the state of the neurons -
        for i ∈ 1:number_of_neurons
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

function _sample(model::MyRestrictedBoltzmannMachineModel, pass::FeedbackPassModel, sₒ::Vector{Int}; 
    T::Int = 100, β::Float = 1.0)


    # initialize -
    W = model.W; # weight matrix
    b = model.b; # hidden bias vector
    a = model.a; # visible bias vector

    number_of_visible_neurons = length(a); # number of visible neurons
    S = zeros(Int, number_of_visible_neurons , T);
    IN = zeros(Float64, number_of_visible_neurons ); # input to the neurons
    is_ok_to_stop = false; # flag to stop the simulation

    # package initial state -
    h = copy(sₒ); # visible state is fixed, sample over the hidden state 
    S[:, 1] .= h; # store the initial state in the S matrix

    # main loop -
    t = 2;
    while (is_ok_to_stop == false)

        # Compute the input to each neuron -
        for i ∈ 1:number_of_hidden_neurons
            IN[i] = dot(W[i,:], h) + a[i]; # compute the input for visible node i -
        end

        # update the state of the neurons -
        for i ∈ 1:number_of_neurons
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

function simulate(model::MyRestrictedBoltzmannMachineModel, pass::Y, sₒ::Vector{Int}; 
    T::Int = 100, β::Float64 = 1.0)::Array{Int,2} where Y <: AbstractPassModel
    return _sample(model, pass, sₒ; T = T, β = β); # multiplex dispatch rocks!!
end


function energy(model::MySimpleBoltzmannMachineModel, s::Vector{Int})::Float64

    # initialize -
    W = model.W; # weight matrix
    b = model.b; # bias vector
    energy = -(1/2)*dot(s, W*s) - dot(b, s); # compute the energy of the state

    # return -
    return energy;
end

function energy(model::MyRestrictedBoltzmannMachine, v::Vector{Int}, h::Vector{Int})::Float64

    # initialize -
    W = model.W; # weight matrix
    b = model.b; # hidden bias vector
    a = model.a; # visible bias vector
    energy = dot(v, W*h) - dot(b, h) - dot(a, v); # compute the energy of the state

    # return -
    return energy;
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