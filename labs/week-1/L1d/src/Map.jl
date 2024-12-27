

function _learn(model::MySimpleSelfOrganizingMapModel, data::Array{<:Number,2}; 
    maxiter::Int = 100, d::Any = Euclidean())
    
    # initialize -
    number_of_neurons = model.number_of_neurons;
    (number_of_samples, number_of_features) = size(data);
    radius = 1.0;
    
    # functions -
    h = model.h; # neighborhood function
    α = model.α; # learning rate function
    σ = model.σ; # neighborhood radius function

    for t ∈ 1:maxiter

        D = rand(1:number_of_samples) |> i -> data[i,:]; # random sample from the data

        # find the best matching unit -
        bmu = argmin([d(D, model.weights[i,:]) for i ∈ 1:number_of_neurons]); # closest neuron to the data point

        # evaluate the neighborhood and learning rate functions
        radius = σ(t, radius);
        learning_rate = α(t);
        
        # update the weights -
        for i ∈ 1:number_of_neurons
            distance = d(model.weights[i,:], model.weights[bmu,:]);
            model.weights[i,:] += learning_rate * h(distance, radius) * (D - model.weights[i,:]);
        end
    end

    # return the weights -
    return weights;
end


function learn(model::T, data::Array{<:Number,2}; 
    maxiter::Int = 100, d::Any = Euclidean()) where T <: MyAbstractDataReductioModel
    return _learn(model, data, maxiter = maxiter, d = d); # multiple dispatch calls the correct method
end