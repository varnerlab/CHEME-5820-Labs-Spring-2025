function solve(model::MyLIFSpikingNeuralNetworkModel, input::Array{Float64}, T::Int64)::Array{Float64, 2}
    
    # initialize: get the model parameters -
    W = model.W; # weight matrix
    ν = model.ν; # membrane potential
    τ = model.τ; # time constant
    Δt = model.Δt; # time step

    # simulation setup
    Tₐ = range(0, stop=T*Δt, step = Δt) |> collect; # time
    number_of_time_steps = length(Tₐ); # number of time steps
    α = exp(-Δt / τ); # decay factor
    output = zeros(Float64, number_of_time_steps, number_of_nodes);
    
    # main loop -
    for i ∈ eachindex(Tₐ)
                

    end

    for i in 1:number_of_nodes
        output[i] = ν[i] + sum(W[i, :] .* input) * Δt / τ;
        output[i] = max(output[i], 0); # apply ReLU activation function
        ν[i] = output[i]; # update membrane potential
    end

    return output;
end