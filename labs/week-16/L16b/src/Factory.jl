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
function build(modeltype::Type{MyDQNLearningAgentModel}, data::NamedTuple)::MyDQNLearningAgentModel

    # initialize -
    model = modeltype(); # build an empty model

    # get data from the NamedTuple -
    γ = data.γ; # discount factor
    α = data.α; # learning rate
    mainnetwork = data.mainnetwork; # main network
    targetnetwork = data.targetnetwork; # target network
    
    # assign data to the model -
    model.γ = γ; # discount factor
    model.α = α; # learning rate
    model.mainnetwork = mainnetwork; # main network
    model.targetnetwork = targetnetwork; # target network

    # return -
    return model;
end