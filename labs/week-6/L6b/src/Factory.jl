
"""
    build(modeltype::Type{MyBinaryWeightedMajorityAlgorithmModel}, 
        data::NamedTuple) -> MyBinaryWeightedMajorityAlgorithmModel

Build a Binary Weighted Majority Algorithm model. This function initializes the model with the given parameters
in the `data` NamedTuple. The model is returned to the caller.

### Arguments
- `modeltype::Type{MyBinaryWeightedMajorityAlgorithmModel}`: the type of the model to build
- `data::NamedTuple`: the parameters to initialize the model

The named tuple `data` must have the following fields:
- `ϵ::Float64`: learning rate
- `n::Int64`: number of experts
- `T::Int64`: number of rounds
- `expert::Function`: expert function
- `adversary::Function`: adversary function
"""
function build(modeltype::Type{MyBinaryWeightedMajorityAlgorithmModel}, 
    data::NamedTuple)::MyBinaryWeightedMajorityAlgorithmModel

    # Initialize - 
    model = modeltype(); # build an empty model
    ϵ = data.ϵ; # learning rate
    n = data.n; # number of experts
    T = data.T; # number of rounds
    expert = data.expert; # expert function
    adversary = data.adversary; # adversary function

    # set the parameters -
    model.ϵ = ϵ;
    model.n = n;
    model.T = T;
    model.expert = expert;
    model.adversary = adversary;
    model.weights = ones(Float64, T+1, n) # initialize the weights array with ones 

    # return the model -
    return model;
end