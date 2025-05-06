"""
    build(modeltype::Type{MyDQNLearningAgentModel}, data::NamedTuple) -> MyDQNLearningAgentModel

Build a DQN learning agent model from a NamedTuple of data. 

### Arguments
    - modeltype::Type{MyDQNLearningAgentModel}: the type of the model to build
    - data::NamedTuple: a NamedTuple containing the data to build the model

The NamedTuple should contain the following fields:
    - γ: discount factor
    - α: learning rate
    - mainnetwork: main network
    - targetnetwork: target network
    - replaybuffer: replay buffer

### Returns
    - model::MyDQNLearningAgentModel: the built model
"""
function build(modeltype::Type{MyDQNLearningAgentModel}, data::NamedTuple)

    # initialize -
    model = modeltype(); # build an empty model

    # get data from the NamedTuple -
    mainnetwork = data.mainnetwork; # main network
    targetnetwork = data.targetnetwork; # target network
    Δ = data.Δ; # perturbation
    K = data.number_of_inputs;
    number_of_actions = data.number_of_actions; # number of actions

    # actions -
    actions::Dict{Int64, Vector{Float32}} = Dict{Int64, Vector{Float32}}(); # actions
    
    # up actions -
    UP = Δ*Matrix{Float32}(I, K, K); # UP is the identity matrix
    linearindex = 1;
    for i ∈ 1:K
        actions[linearindex] = UP[:,i]; # assign the UP action
        linearindex += 1;
    end
    
    # down actions -
    DOWN = -Δ*Matrix{Float32}(I, K, K); # DOWN is the identity matrix
    for i ∈ 1:K
        actions[linearindex] = DOWN[:,i]; # assign the UP action
        linearindex += 1;
    end

    # assign data to the model -
    model.mainnetwork = mainnetwork; # main network
    model.targetnetwork = targetnetwork; # target network
    model.actions = actions; # actions
    model.number_of_actions = number_of_actions; # number of actions
    model.number_of_inputs = K; # number of inputs

    # return -
    return model;
end

function build(modeltype::Type{MyDQNworldContextModel}, data::NamedTuple)::MyDQNworldContextModel

    # initialize -
    m = data.m; # number of arms categories of goods
    γ = data.γ; # consumer's preference for each category of goods
    σ = data.σ; # uncetainty of consumer's preference for each category of goods
    Z = data.Z; # consumer's error model
    C = data.C; # price of each good in each category
    λ = data.λ; # how budget sensitive the consumer is
    B = data.B; # consumer's budget

    # build empty model -
    model = modeltype();
    model.m = m;
    model.γ = γ;
    model.σ = σ;
    model.Z = Z;
    model.C = C;
    model.λ = λ;
    model.B = B;

    # return -
    return model;
end