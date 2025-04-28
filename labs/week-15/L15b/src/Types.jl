abstract type AbstractSpikingNeuralNetworkModel end
abstract type AbstractStateSpaceSequenceModel end


mutable struct MyLIFSpikingNeuralNetworkModel <: AbstractSpikingNeuralNetworkModel

    # model parameters -
    number_of_nodes::Int64; # number of nodes in the network
    number_of_inputs::Int64; # number of inputs to the network
    W::Matrix{Float64}; # weight matrix
    ν::Float64; # membrane potential
    τ::Float64; # time constant
    Δt::Float64; # time step

    # empty constructor -
    MyLIFSpikingNeuralNetworkModel() = new();
end

mutable struct MyS5Model <: AbstractStateSpaceSequenceModel

    # model parameters -
    number_of_output::Int64; # number of outputs from the model 
    number_of_inputs::Int64; # number of inputs into the network
    number_of_hidden_states::Int64; # number of hidden states
    Δt::Float64; # time step
    Ā::Matrix{Float64}; # state transition matrix
    B̄::Matrix{Float64}; # input matrix
    C̄::Matrix{Float64}; # output matrix
    D̄::Matrix{Float64}; # feedforward matrix
    
    # empty constructor -
    MyS5Model() = new();
end