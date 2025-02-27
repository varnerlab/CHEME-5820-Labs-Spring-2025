function _children(edges::Dict{Tuple{Int, Int}, Tuple{Float64, Float64, Float64}}, id::Int64)::Set{Int64}
    
    # initialize -
    childrenset = Set{Int64}();

    # Dumb implementation - why?
    for (k, _) ∈ edges
        if k[1] == id
            push!(childrenset, k[2]);
        end
    end

    # return -
    return childrenset;
end

function log_growth_matrix(dataset::Dict{String, DataFrame}, 
    firms::Array{String,1}; Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0)::Array{Float64,2}

    # initialize -
    number_of_firms = length(firms);
    number_of_trading_days = nrow(dataset["AAPL"]);
    return_matrix = Array{Float64,2}(undef, number_of_trading_days-1, number_of_firms);

    # main loop -
    for i ∈ eachindex(firms) 

        # get the firm data -
        firm_index = firms[i];
        firm_data = dataset[firm_index];

        # compute the log returns -
        for j ∈ 2:number_of_trading_days
            S₁ = firm_data[j-1, :volume_weighted_average_price];
            S₂ = firm_data[j, :volume_weighted_average_price];
            return_matrix[j-1, i] = (1/Δt)*log(S₂/S₁) - risk_free_rate;
        end
    end

    # return -
    return return_matrix;
end

"""
    function children(graph::T, node::MyGraphNodeModel) -> Set{Int64} where T <: MyAbstractGraphModel

This function returns the children of a node in a graph model.

### Arguments
- `graph::T`: the graph model to search. This is a subtype of `MyAbstractGraphModel`.
- `node::MyGraphNodeModel`: the node to search for children.

### Returns
- a set of children node ids.
"""
function children(graph::T, node::MyGraphNodeModel)::Set{Int64} where T <: MyAbstractGraphModel
    return graph.children[node.id];
end

"""
    function weight(graph::T, source::Int64, target::Int64) -> Float64 where T <: MyAbstractGraphModel

This function returns the weight of the edge between two nodes in a graph model.

### Arguments
- `graph::T`: the graph model to search. This is a subtype of `MyAbstractGraphModel`.
- `source::Int64`: the source node id.
- `target::Int64`: the target node id.

### Returns
- the weight of the edge between the source and target nodes.
"""
function weight(graph::T, source::Int64, target::Int64)::Float64 where T <: MyAbstractGraphModel   
    return graph.edges[(source, target)][1];
end