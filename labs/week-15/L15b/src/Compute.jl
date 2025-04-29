
H(x,ν) = x ≥ ν ? 1 : 0 # Heaviside step function

"""
    solve(model::MyLIFSpikingNeuralNetworkModel, input::Array{Int}; 
        T::Int64 = 100, Δᵣ::Int = 3, sₒ::Array{Int64} = nothing) -> Tuple{Array{Float64}, Array{Float64}, Array{Float64}}

Runs a LIF spiking neural network model for T time steps with a given input.

### Arguments
- `model::MyLIFSpikingNeuralNetworkModel`: The LIF spiking neural network model to run.
- `input::Array{Int}`: The input to the model, a 2D array where each column is a time step.
- `T::Int64`: The number of time steps to run the model for.
- `Δᵣ::Int`: The refractory period for the neurons in the model.
- `sₒ::Array{Int64}`: The initial spike train for the neurons in the model.

### Returns
- `Tuple{Array{Float64}, Array{Float64}, Array{Float64}}`: A tuple containing:
    - The time steps of the simulation.
    - The membrane potential of the neurons at each time step.
    - The spike train of the neurons at each time step.
"""
function solve(model::MyLIFSpikingNeuralNetworkModel, input::Array{Int}; 
    T::Int64 = 100, Δᵣ::Int = 3, sₒ::Array{Int64} = nothing)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}}
    
    # initialize: get the model parameters -
    W = model.W; # weight matrix
    ν = model.ν; # membrane potential
    τ = model.τ; # time constant
    Δt = model.Δt; # time step
    number_of_nodes = model.number_of_nodes; # number of nodes in the network

    # simulation setup
    Tₐ = range(1, stop=T*Δt, step = Δt) |> collect; # time
    number_of_time_steps = length(Tₐ); # number of time steps
    α = exp(-Δt / τ); # decay factor
    s = zeros(Float64, number_of_time_steps+1, number_of_nodes); # spike train 
    V = ν*rand(Float64, number_of_time_steps+1, number_of_nodes); # membrane potential is initially zero?
    history = Dict{Int, Int}(); # history of the neuron spikes

    # check: do we have an initial spike train?
    if isnothing(sₒ) == false
        for i ∈ 1:number_of_nodes
            s[1,i] = sₒ[i]; # set the initial spike train
        end
    end


    # initialize the history to 0 -
    for i ∈ 1:number_of_nodes
        history[i] = 0; # no spikes yet
    end
    
    # main loop -
    for i ∈ eachindex(Tₐ)
        z = input[:,i]; # input at time step i = from pre-synaptic neurons
        
        for j ∈ 1:number_of_nodes
            
            # compute the input to neuron j -
            I = dot(W[j, :], z); # input current
            V[i+1, j] = α*V[i,j] + (1 - α)*I - ν*s[i,j]; # update the membrane potential

            # can this neuron spike?
            dt = history[j]; # time since last spike
            if (H(V[i+1, j],ν) == 1 && dt == 0) # the neuron spikes, and not in refractory period
                s[i+1, j] = 1; # set the spike train to 1
                history[j] = Δᵣ; # set the reactory period
            elseif (dt > 0) # if the neuron does not spike
                history[j] = dt - 1; # decrement the time since last spike
            end            
        end
    end

    # return -
    return  Tₐ, V, s[2:end, :]; # return the time, membrane potential, and spike train
end

"""
    solve(m::MyS5Model, input::Array{Float64}, T::Int64) -> Tuple{Array{Float64}, Array{Float64}, Array{Float64}}

Runs an S5 (simplified structured state space sequence) model for T time steps with a given input.

### Arguments
- `m::MyS5Model`: The S5 model to run.
- `input::Array{Float64}`: The input to the model, a 2D array where each column is a time step.
- `T::Int64`: The number of time steps to run the model for.

### Returns
- `Tuple{Array{Float64}, Array{Float64}, Array{Float64}}`: A tuple containing:
    - The time steps of the simulation.
    - The hidden state of the model at each time step.
    - The output of the model at each time step.
"""
function solve(m::MyS5Model, input::Array{Float64}, T::Int64)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}}
    
    # initialize: get the model parameters -
    Ā = m.Ā; # state transition matrix
    B̄ = m.B̄; # input matrix
    C̄ = m.C̄; # output matrix
    D̄ = m.D̄; # feedforward matrix
    Δt = m.Δt; # time step
    number_of_hidden_states = m.number_of_hidden_states; # number of hidden states
    number_of_outputs = m.number_of_output; # number of outputs from the model

    # setup stuff required for the simulation -
    Tₐ = range(1, stop=T*Δt, step = Δt) |> collect; # time
    number_of_time_steps = length(Tₐ); # number of time steps
    H = zeros(Float64, number_of_time_steps, number_of_hidden_states); # hidden state
    Y = zeros(Float64, number_of_time_steps, number_of_outputs); # output

    # main loop -
    for i ∈ eachindex(Tₐ)
        
        uᵢ = input[:,i]; # input at time step i

        if (i == 1)
            H[i, :] = B̄*uᵢ; # initial hidden state
            Y[i, :] = C̄*H[i, :] + D̄*uᵢ; # initial output
        else
            H[i, :] = Ā*H[i-1, :] + B̄*uᵢ; # update the hidden state
            Y[i, :] = C̄*H[i, :] + D̄*uᵢ; # update the output
        end

        # check for negatives -
        if (any(H[i, :] .< 0))
            H[i,:] = max.(H[i,:], 0); # set negative values to zero
        end
    end

    # return -
    return Tₐ, H, Y;
end

# fun hack!
#(m::MyS5Model)(input::Array{T<:Number}, T::Int64)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}} = _s5solve(m, input, T)
#(m::MyLIFSpikingNeuralNetworkModel)(input::Array{Int}, T::Int64; Δ::Int = 3)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}} =  solve(m, input, T, Δ = Δ)