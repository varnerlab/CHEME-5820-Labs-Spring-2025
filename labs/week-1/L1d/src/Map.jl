

function _learn(model::MySimpleSelfOrganizingMapModel, data::Array{<:Number,2}; 
    maxiter::Int = 100, d::Any = Euclidean())
    
    # initialize -
    number_of_neurons = model.number_of_neurons;
    number_of_features = model.number_of_features;
    h = model.h;
    α = model.α;
    σ = model.σ;

    for t ∈ 1:maxiter
        for i ∈ 1:size(data, 2)
            
            # find the best matching unit -
            x = data[:,i];
            bmu = argmin([d(x, model.weights[:,j]) for j ∈ 1:number_of_neurons]);
            
            # update the weights -
            for j ∈ 1:number_of_neurons
                model.weights[:,j] += model.α(t) * model.h(d(bmu, j, t)) * (x - model.weights[:,j]);
            end
        end
    end


end


function learn(model::T, data::Array{<:Number,2}; 
    maxiter::Int = 100, d::Any = Euclidean()) where T <: MyAbstractDataReductionModel
    return _learn(model, data, maxiter = maxiter, d = d); # multiple dispatch calls the correct method
end