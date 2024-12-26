

function _learn(model::MySimpleSelfOrganizingMapModel, data::Array{<:Number,2}; 
    maxiter::Int = 100)
    
    # ...
end


function learn(model::T, data::Array{<:Number,2}; 
    maxiter::Int = 100) where T<:MyAbstractDataReduction
    return _learn(model, data, maxiter = maxiter); # multiple dispatch calls the correct method
end