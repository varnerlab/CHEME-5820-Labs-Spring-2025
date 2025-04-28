function build(modeltype::Type{MyLIFSpikingNeuralNetworkModel}, data::NamedTuple)::MyLIFSpikingNeuralNetworkModel

    # build an empty model -
    model = modeltype();

    # get stuff from the data -
    Δt = data.Δt; # time step
    τ = data.τ; # time constant
    number_of_nodes = data.number_of_nodes; # number of nodes in the network
    W = data.W; # weight matrix
    ν = data.ν; # membrane potential

    # set the model parameters -
    model.Δt = Δt; # time step
    model.τ = τ; # time constant
    model.number_of_nodes = number_of_nodes; # number of nodes in the network
    model.W = W; # weight matrix
    model.ν = ν; # membrane potential
    model.number_of_inputs = size(W, 2); # number of inputs to the network

    # return -
    return model;
end