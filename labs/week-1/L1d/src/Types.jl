abstract type MyAbstractDataReductioModel end

mutable struct MySimpleRectangularSelfOrganizingMapModel <: MyAbstractDataReductioModel
    # data -
    number_of_neurons::Int
    weights::Array{Float64,2}; # weights has the shape of (number_of_neurons, number_of_features)
    neurons::Dict{Int, Tuple{Int,Int}} # states of the neurons
    coordinates::Dict{Tuple{Int,Int}, Int} # coordinates of the neurons

    # behavior -
    h::Function # neighborhood function
    α::Function # learning rate function
    σ::Function # neighborhood radius function

    # constructor -
    MySimpleRectangularSelfOrganizingMapModel() = new();
end