abstract type AbstractOnlineLearningModel end # super type for all online learning models


mutable struct MyBinaryWeightedMajorityAlgorithmModel <: AbstractOnlineLearningModel
    
    # parameters
    ϵ::Float64 # learning rate
    n::Int64 # number of experts
    T::Int64 # number of rounds
    weights::Array{Float64,2} # weights of the experts

    # default constructor -
    MyBinaryWeightedMajorityAlgorithmModel() = new();
end


