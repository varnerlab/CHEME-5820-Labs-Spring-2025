
function build(modeltype::Type{MySimpleRectangularSelfOrganizingMapModel}, data::NamedTuple)::MySimpleRectangularSelfOrganizingMapModel

    # build an empty model -
    model = modeltype(); # this constructs an empty model, we need to fill it with data

    # initialize -
    number_of_neurons = data.number_of_neurons;
    number_of_features = data.number_of_features;
    number_of_nodes = sqrt(number_of_neurons);
    
    # functions -
    h = data.h; # neighborhood function
    α = data.α; # learning rate function
    σ = data.σ; # neighborhood radius function

    # weights -
    weights = randn(number_of_neurons, number_of_features);

    # build the neurons dictionary -
    neurons = Dict{Int, Tuple{Int,Int}}();
    linearindex = 1;
    for i ∈ 1:number_of_nodes
        for j ∈ 1:number_of_nodes
            neurons[linearindex] = (i,j);
            linearindex += 1;
        end
    end

    # build the coordinates dictionary -
    coordinates = Dict{Tuple{Int,Int}, Int}();
    for (k, v) ∈ neurons
        coordinates[v] = k;
    end


    # fill the model -
    model.number_of_neurons = number_of_neurons;
    model.weights = weights;
    model.h = h;
    model.α = α;
    model.σ = σ;
    model.neurons = neurons;
    model.coordinates = coordinates;


    # return the model -
    return model;
end