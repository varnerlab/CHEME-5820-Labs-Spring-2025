abstract type AbstractSimpleChoiceProblem end
abstract type AbstractReturnModel end

"""
    mutable MySimpleCobbDouglasChoiceProblem

A mutable struct that defines a simple Cobb-Douglas choice problem. 
The problem is defined by the following fields:

### Fields
- `γ::Array{Float64,1}`: The Cobb-Douglas utility function exponents. One exponent for each object.
- `c::Array{Float64,1}`: The unit cost of each objects.
- `B::Float64`: The budget that we have to spend on the collection of objects.
- `bounds::Array{Float64,2}`: The bounds on the objects that we can purchase. First column is the lower bound, second column is the upper bound.
- `initial::Array{Float64,1}`: The initial guess for the solution.

"""
mutable struct MySimpleCobbDouglasChoiceProblem <: AbstractSimpleChoiceProblem

    # data -
    γ::Array{Float64,1}
    c::Array{Float64,1}
    B::Float64
    bounds::Array{Float64,2}
    initial::Array{Float64,1}

    # constructor
    MySimpleCobbDouglasChoiceProblem() = new();
end

"""
    mutable struct MySingleIndexModel <: AbstractReturnModel

The `MySingleIndexModel` mutable struct represents a single index model of risky asset returns.

### Required fields
- `α::Float64`: The firm specific unexplained return
- `β::Float64`: The relationship between the firm and the market
- `r::Float64`: The risk-free rate of return
- `ϵ::Distribution`: The random shocks to the model (unexplained return)    
"""
mutable struct MySingleIndexModel <: AbstractReturnModel

    # model -
    α::Float64          # firm specific unexplained return
    β::Float64          # relationship between the firm and the market
    r::Float64          # risk free rate of return 
    ϵ::Distribution     # random shocks 

    # constructor -
    MySingleIndexModel() = new()
end