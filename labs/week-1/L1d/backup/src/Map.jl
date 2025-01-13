

function _learn(model::MySimpleRectangularSelfOrganizingMapModel, data::Array{<:Number,2}; 
    maxiter::Int = 100, d::Any = Euclidean(), ϵ::Float64 = 0.01)
    
    # initialize -
    number_of_neurons = model.number_of_neurons;
    (number_of_samples, number_of_features) = size(data);
    radius = 10.0;
    t = 1;
    
    # functions -
    h = model.h; # neighborhood function
    α = model.α; # learning rate function
    σ = model.σ; # neighborhood radius function

    should_stop_looping = false;
    while (should_stop_looping == false)

        D = rand(1:number_of_samples) |> i -> data[i,:]; # random sample from the data

        # find the best matching unit -
        bmu = argmin([d(D, model.weights[i,:]) for i ∈ 1:number_of_neurons]); # closest neuron to the data point

        # evaluate the neighborhood and learning rate functions
        radius = σ(t, radius);
        learning_rate = α(t);
        
        # update the weights -
        Ŵ = copy(model.weights);
        for i ∈ 1:number_of_neurons
            distance = d(model.weights[i,:], model.weights[bmu,:]);
            model.weights[i,:] += learning_rate * h(distance, radius) * (D - model.weights[i,:]);
        end

        #  weights should be positive, since all the data is positive -
        # model.weights = max.(model.weights, 0.0);

        # check if we should stop looping -
        if (t >= maxiter || opnorm(Ŵ - model.weights) < ϵ)
            should_stop_looping = true;

            println("Stopping at iteration: $(t)");

        else
            t += 1;
        end
    end

   
    # return the weights -
    return model.weights;
end


function learn(model::T, data::Array{<:Number,2}; 
    maxiter::Int = 100, d::Any = Euclidean(), ϵ::Float64 = 0.01) where T <: MyAbstractDataReductioModel
    return _learn(model, data, maxiter = maxiter, d = d, ϵ = ϵ); # multiple dispatch calls the correct method
end