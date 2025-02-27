abstract type AbstractOnlineLearningModel end # super type for all online learning models
abstract type MyAbstractGraphModel end
abstract type MyAbstractGraphNodeModel end
abstract type MyAbstractGraphEdgeModel end

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

mutable struct MyTwoPersonZeroSumGameModel <: AbstractOnlineLearningModel
    
    # parameters
    ϵ::Float64 # learning rate
    n::Int64 # number of experts (actions)
    T::Int64 # number of rounds
    weights::Array{Float64,2} # weights of the experts
    payoffmatrix::Array{Float64,2} # payoff matrix

    # default constructor -
    MyTwoPersonZeroSumGameModel() = new();
end

mutable struct MyConstraintCheckingGameModel <: AbstractOnlineLearningModel
    
    # parameters
    η::Float64 # learning rate
    T::Int64 # number of rounds
    A::Array{Float64,2} # constraint matrix
    b::Array{Float64,1} # right hand side vector
    τ::Float64 # summation of x upper bound
    ρ::Float64 # upper/lower of the elements of the matrix A
    weights::Array{Float64,2} # weights of the experts

    # default constructor -
    MyConstraintCheckingGameModel() = new();
end

mutable struct MyGraphNodeModel <: MyAbstractGraphNodeModel
   
    # data -
    id::Int64
 
    # constructor -
    MyGraphNodeModel(id::Int64) = new(id);
 end
 
 mutable struct MyGraphEdgeModel <: MyAbstractGraphEdgeModel
    
    # data -
    id::Int64
    source::Int64
    target::Int64
    cost::Union{Nothing, Float64}; # this is a little fancy??
    lower_bound_capacity::Union{Nothing, Float64}; # this is a little fancy??
    upper_bound_capacity::Union{Nothing, Float64}; # this is a little fancy??
 
    # constructor -
    MyGraphEdgeModel() = new();
 end
 
 mutable struct MySimpleDirectedGraphModel <: MyAbstractGraphModel
    
   # data -
   nodes::Union{Nothing, Dict{Int64, MyGraphNodeModel}}
   edges::Union{Nothing, Dict{Tuple{Int, Int}, Tuple{Float64, Float64, Float64}}}; # first Float64 is the cost, second Float64 is the capacity
   edgesinverse::Dict{Int, Tuple{Int, Int}} # map between edge id and source and target
   children::Union{Nothing, Dict{Int64, Set{Int64}}}
   A::Array{Float64,2}; # system constraint matrix for flow optimization
 
   # constructor -
   MySimpleDirectedGraphModel() = new();
 end