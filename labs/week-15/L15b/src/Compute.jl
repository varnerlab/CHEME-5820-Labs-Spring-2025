
H(x,ν) = x ≥ ν ? 1 : 0 # Heaviside step function

function solve(model::MyLIFSpikingNeuralNetworkModel, 
    input::Array{Int}, T::Int64; Δ::Int = 3)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}}
    
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
            
            if (H(V[i+1, j],ν) == 1 && dt == 0) # if the neuron spikes
                s[i+1, j] = 1; # set the spike train to 1
                history[j] = Δ; # set the reactory period
            elseif (dt > 0) # if the neuron does not spike
                history[j] = dt - 1; # decrement the time since last spike
            end            
        end
    end

    # return -
    return  Tₐ, V, s;
end

# fun hack!
(m::MyLIFSpikingNeuralNetworkModel)(input::Array{Int}, T::Int64; Δ::Int = 3)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}} =  solve(m, input, T, Δ = Δ)