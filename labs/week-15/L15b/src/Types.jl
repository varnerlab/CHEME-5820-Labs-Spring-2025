abstract type AbstractSpikingNeuralNetworkModel end


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