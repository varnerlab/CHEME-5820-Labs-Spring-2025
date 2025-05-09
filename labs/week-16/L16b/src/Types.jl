abstract type AbstractNeuralNetwork end
abstract type AbstractLearningModel end
abstract type AbstractWorldContextModel end

# build a simple neural network model type -
struct MyFluxNeuralNetworkModel <: AbstractNeuralNetwork
    chain::Chain; # holds the model chain
end

"""
    mutable struct MyDQNLearningAgentModel <: AbstractLearningModel

A mutable struct to hold the DQN learning agent model.

### Fields
    - actions::Dict{Int64, Vector{Float32}}: the actions of the agent
    - mainnetwork::Chain: the main network of the agent
    - targetnetwork::Chain: the target network of the agent
    - number_of_actions::Int64: the number of actions of the agent
    - number_of_inputs::Int64: the number of inputs of the agent
    - replaybuffer::CircularBuffer{Tuple{Vector{Float32},Vector{Float32}, Float32, Vector{Float32}}}: the replay buffer of the agent
"""
mutable struct  MyDQNLearningAgentModel<: AbstractLearningModel

    # initialization -
    actions::Dict{Int64, Vector{Float32}} # actions
    mainnetwork::Chain # main network
    targetnetwork::Chain # target network
    number_of_actions::Int64 # number of actions
    number_of_inputs::Int64 # number of inputs
    replaybuffer::CircularBuffer{Tuple{Vector{Float32},Vector{Float32}, Float32, Vector{Float32}}} # replay buffer

    # empty constructor -
    MyDQNLearningAgentModel() = new();
end

"""
    mutable struct MyDQNworldContextModel <: AbstractWorldContextModel

A mutable struct to hold the context of the world model.

### Fields
    - m::Int64: the number of goods to choose from
    - γ::Array{Int64,1}: consumer's preference for each category of goods
    - σ::Array{Float32,2}: uncertainty of consumer's preference for each category of goods, and the price of each good in each category
    - C::Array{Float32,1}: price of each good in each category
    - λ::Float64: how budget sensitive the consumer is
    - B::Float64: consumer's budget
    - Z::Normal: consumer's preference for each category of goods
"""
mutable struct MyDQNworldContextModel <: AbstractWorldContextModel

    # data -
    m::Int64 # number of of goods to choose from
    γ::Array{Int64,1} # consumer's preference for each category of goods
    σ::Array{Float32,2} # uncetainty of consumer's preference for each category of goods, and the price of each good in each category
    C::Array{Float32,1} # price of each good in each category
    λ::Float64 # how budget sensitive the consumer is
    B::Float64 # consumer's budget
    Z::Normal # consumer's preference for each category of goods

    # constructor -
    MyDQNworldContextModel() = new();
end