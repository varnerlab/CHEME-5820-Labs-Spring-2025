abstract type MyAbstractDataReductioModel end

mutable struct MySimpleSelfOrganizingMapModel <: MyAbstractDataReductionModel

    # data -
    number_of_neurons::Int
    number_of_features::Int
    number_of_examples::Int
    weights::Array{Float64,2}

    # behavior -
    h::Function # neighborhood function
    α::Function # learning rate function
    σ::Function # neighborhood radius function

    # constructor -
    MySimpleSelfOrganizingMapModel() = new();
end