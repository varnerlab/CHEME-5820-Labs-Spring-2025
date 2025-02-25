abstract type AbstractOnlineLearningModel end # super type for all online learning models

"""
    MyBinaryWeightedMajorityAlgorithmModel

A mutable type for the Binary Weighted Majority Algorithm model. 
This model is used to simulate the Binary Weighted Majority Algorithm. The model has the following fields:

- `ϵ::Float64`: learning rate
- `n::Int64`: number of experts
- `T::Int64`: number of rounds
- `weights::Array{Float64,2}`: weights of the experts
- `expert::Function`: expert function
- `adversary::Function`: adversary function
"""
mutable struct MyBinaryWeightedMajorityAlgorithmModel <: AbstractOnlineLearningModel
    
    # parameters
    ϵ::Float64 # learning rate
    n::Int64 # number of experts
    T::Int64 # number of rounds
    weights::Array{Float64,2} # weights of the experts
    expert::Function # expert function
    adversary::Function # adversary function

    # default constructor -
    MyBinaryWeightedMajorityAlgorithmModel() = new();
end


