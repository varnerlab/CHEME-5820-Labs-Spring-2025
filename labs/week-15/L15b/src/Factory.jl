
"""
    build(modeltype::Type{MyLIFSpikingNeuralNetworkModel}, data::NamedTuple) -> MyLIFSpikingNeuralNetworkModel

Factory function to build a MyLIFSpikingNeuralNetworkModel object.

### Arguments
- `modeltype::Type{MyLIFSpikingNeuralNetworkModel}`: The type of the model to build.
- `data::NamedTuple`: A named tuple containing the model parameters. The named tuple should
  contain the following fields:
    - `number_of_nodes`: The number of nodes in the network.
    - `number_of_inputs`: The number of inputs to the network.
    - `Δt`: The time step.
    - `τ`: The time constant.
    - `ν`: The membrane potential.

### Returns
- a `MyLIFSpikingNeuralNetworkModel` object with the model parameters set.
"""
function build(modeltype::Type{MyLIFSpikingNeuralNetworkModel}, data::NamedTuple)::MyLIFSpikingNeuralNetworkModel

    # build an empty model -
    model = modeltype();

    # get stuff from the data -
    Δt = data.Δt; # time step
    τ = data.τ; # time constant
    number_of_nodes = data.number_of_nodes; # number of nodes in the network
    number_of_inputs = data.number_of_inputs; # number of inputs to the network
    ν = data.ν; # membrane potential

    # build an initial W -
    W = (ν^2)*rand(Float64, number_of_nodes, number_of_inputs); # weight matrix

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

"""
    build(modeltype::Type{MyS5Model}, data::NamedTuple) -> MyS5Model

Factory function to build a MyS5Model object.

### Arguments
- `modeltype::Type{MyS5Model}`: The type of the model to build.
- `data::NamedTuple`: A named tuple containing the model parameters. The named tuple should
  contain the following fields:
    - `number_of_inputs`: The number of inputs to the network.
    - `number_of_outputs`: The number of outputs from the model.
    - `number_of_hidden_states`: The number of hidden states.
    - `Δt`: The time step.

### Returns
- a `MyS5Model` object with the discretized state space model parameters set.
"""
function build(modeltype::Type{MyS5Model}, data::NamedTuple)::MyS5Model

    # build an empty model -
    model = modeltype();

    # get stuff from the data -
    number_of_inputs = data.number_of_inputs; # number of inputs to the network
    number_of_outputs = data.number_of_outputs; # number of outputs from the model
    number_of_hidden_states = data.number_of_hidden_states; # number of hidden states
    Δt = data.Δt; # time step
    
    # build A -
    AN = Array{Float64,2}(undef, number_of_hidden_states, number_of_hidden_states); # internal hidden state memory
    P = Array{Float64,2}(undef, number_of_hidden_states, 1); # low rank approximation
    for i ∈ 1:number_of_hidden_states
        
        P[i,1] = sqrt((i+1/2));
        for k = 1:number_of_hidden_states
            
            if (i > k)
                AN[i,k] = -sqrt((i+1/2))*sqrt((k+1/2));
            elseif (i == k)
                AN[i,k] = -1/2;
            else
                AN[i,k] = -sqrt((i+1/2))*sqrt((k+1/2));
            end
        end
    end
    A = AN - P*P'; # A = A + P*P'
    V = eigen(A) |> F -> F.vectors;   # eigenvectors
    
    # Build B -
    B = Array{Float64,2}(undef, number_of_hidden_states, number_of_inputs); # input matrix
    for i ∈ 1:number_of_inputs
        for j ∈ 1:number_of_hidden_states
            B[j,i] = sqrt(2*j+1); # we just repeat bⱼ across the columns
        end
    end

    # Build C and D -
    C = randn(Float64, number_of_outputs, number_of_hidden_states); # output matrix
    D = zeros(Float64, number_of_outputs, number_of_inputs); # feedforward matrix

    # Rotate -
    Λ = inv(V)*A*V |> M-> round.(M, digits = 4); # diagonalize A
    B̃ = (1/number_of_inputs)*inv(V)*B;
    C̃ = C*V;
    D̃ = D*V;

    # discritize (bilinear) -
    IM = Matrix{Float64}(I, number_of_hidden_states, number_of_hidden_states); # identity matrix
    Ā = inv(IM - (Δt/2)*Λ)*(IM + (Δt/2)*Λ); # state transition matrix
    B̄ = inv(IM - (Δt/2)*Λ)*(Δt)*B̃; # input matrix
    C̄ = C̃;
    D̄ = D̃;
    # Ā = exp(Λ*Δt); # state transition matrix
    # B̄ = inv(Λ)*(IM - Λ)*B̃; # input matrix
    # C̄ = C̃;
    # D̄ = D̃;


    # set parameters on the model -
    model.Δt = Δt; # time step
    model.number_of_inputs = number_of_inputs; # number of inputs to the network
    model.number_of_output = number_of_outputs; # number of outputs from the model
    model.number_of_hidden_states = number_of_hidden_states; # number of hidden states
    model.Ā = Ā; # state transition matrix
    model.B̄ = B̄; # input matrix
    model.C̄ = C̄; # output matrix
    model.D̄ = D̄; # feedforward matrix


    # return -
    return model;
end