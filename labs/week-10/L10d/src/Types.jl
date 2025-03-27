abstract type AbstractBoltzmannMachineModel end
abstract type AbstractPassModel end

struct MyFeedForwardPassModel <: AbstractPassModel
end

struct MyFeedbackPassModel <: AbstractPassModel
end

mutable struct MySimpleBoltzmannMachineModel <: AbstractBoltzmannMachineModel
    
    # fields
    W::Array{Float64,2}; # weight matrix
    b::Vector{Float64}; # bias vector

    # constructor
    MySimpleBoltzmannMachineModel() = new();
end

mutable struct MyRestrictedBoltzmannMachineModel  <: AbstractBoltzmannMachineModel

    # fields -
    W::Array{Float64,2}; # weight matrix
    b::Vector{Float64}; # hidden bias vector
    a::Vector{Float64}; # visible bias vector

    # constructor -
    MyRestrictedBoltzmannMachineModel() = new();
end