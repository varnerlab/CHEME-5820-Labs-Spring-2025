
function build(modeltype::Type{MySimpleSelfOrganizingMapModel}, data::NamedTuple)::MySimpleSelfOrganizingMapModel

    # build an empty model -
    model = modeltype(); # this constructs an empty model, we need to fill it with data

    # initialize -
    number_of_neurons = data.number_of_neurons;
    number_of_features = data.number_of_features;
    
    # functions -
    h = data.h; # neighborhood function
    α = data.α; # learning rate function
    σ = data.σ; # neighborhood radius function

    # weights -
    weights = randn(number_of_neurons, number_of_features);

    # fill the model -
    model.number_of_neurons = number_of_neurons;
    model.weights = weights;
    model.h = h;
    model.α = α;
    model.σ = σ;

    # return the model -
    return model;
end