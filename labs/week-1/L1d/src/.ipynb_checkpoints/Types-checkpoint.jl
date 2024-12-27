abstract type MyAbstractDataReductioModel end

mutable struct MySimpleSelfOrganizingMapModel <: MyAbstractDataReductionModel

    # data -
    number_of_neurons::Int
    weights::Array{Float64,2}; # weights has the shape of (number_of_neurons, number_of_features)

    # behavior -
    h::Function # neighborhood function
    α::Function # learning rate function
    σ::Function # neighborhood radius function

    # constructor -
    MySimpleSelfOrganizingMapModel() = new();
end