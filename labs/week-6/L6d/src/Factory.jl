function _build(edgemodel::Type{MyGraphEdgeModel}, parts::Array{String,1}, id::Int64)::MyGraphEdgeModel
    
    # initialize -
    model = edgemodel(); # build an empty edge model
    
    # populate -
    model.id = id;
    model.source = parse(Int64, parts[1]);
    model.target = parse(Int64, parts[2]);
    model.cost = parse(Float64, parts[3]);
    model.lower_bound_capacity = parse(Float64, parts[4]);
    model.upper_bound_capacity = parse(Float64, parts[5]);

    # return -
    return model
end


function build(modeltype::Type{MyBinaryWeightedMajorityAlgorithmModel}, 
    data::NamedTuple)

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

function build(modeltype::Type{MyTwoPersonZeroSumGameModel},
    data::NamedTuple)

    # initialize -
    model = modeltype(); # build an empty model
    ϵ = data.ϵ; # learning rate
    n = data.n; # number of experts (actions)
    T = data.T; # number of rounds
    payoffmatrix = data.payoffmatrix; # payoff matrix

    # set the parameters -
    model.ϵ = ϵ;
    model.n = n;
    model.T = T;
    model.payoffmatrix = payoffmatrix;
    model.weights = zeros(Float64, T+1, n) # initialize the weights array with ones

    # generate a random initial weight vector -
    model.weights[1, :] = rand(n);

    # return the model -
    return model;
end

function build(modeltype::Type{MyConstraintCheckingGameModel},
    data::NamedTuple)

    # initialize -
    model = modeltype(); # build an empty model
    η = data.η; # learning rate
    T = data.T; # number of rounds
    A = data.A; # constraint matrix
    b = data.b; # constraint vector
    ρ = data.ρ; # upper/lower of the elements of the matrix A
    τ = data.τ; # summation of x upper bound

    n = size(A, 1); # number of constraints
    m = size(A, 2); # number of variables
    w = zeros(Float64, T+1, m) # one expert per variable
    w[1,:] .= 1; # initialize to 1

    # set the parameters -
    model.η = η;
    model.T = T;
    model.A = A;
    model.b = b;
    model.ρ = ρ;
    model.τ = τ;
    model.weights = w # initialize the weights array with one , one expert for each constraint

    # return -
    return model;
end


# --- PUBLIC METHODS BELOW HERE ----------------------------------------------------------------------------------- #
"""
    build(model::Type{T}, parts::Array{String,1}, id::Int64) where T <: MyGraphEdgeModel
"""
function build(model::Type{T}, edgemodels::Dict{Int64, MyGraphEdgeModel}) where T <: MyAbstractGraphModel

    # build and empty graph model -
    graphmodel = model();
    nodes = Dict{Int64, MyGraphNodeModel}();
    edges = Dict{Tuple{Int64, Int64}, Tuple{Float64,Float64, Float64}}();
    edgesinverse = Dict{Int, Tuple{Int, Int}}();
    children = Dict{Int64, Set{Int64}}();

    # -- DO STUFF WITH NODES -------------------------------------------------- #
    # let's build a list of nodes ids -
    tmp_node_ids = Set{Int64}();
    for (_,v) ∈ edgemodels
        push!(tmp_node_ids, v.source);
        push!(tmp_node_ids, v.target);
    end
    list_of_node_ids = tmp_node_ids |> collect |> sort;

    # build the nodes models with the id's
    [nodes[id] = MyGraphNodeModel(id) for id ∈ list_of_node_ids];
    # --------------------------------------------------------------------------- #
    
    # -- DO STUFF WITH EDGES ---------------------------------------------------- #
    # build the edges dictionary (source, target) -> (cost, lower_bound_capacity, upper_bound_capacity
    for (_, v) ∈ edgemodels
        source_index = v.source;
        target_index = v.target;
        edges[(source_index, target_index)] = (v.cost, v.lower_bound_capacity, v.upper_bound_capacity);
    end

    # build the inverse edge dictionary edgeid -> (source, target)
    n = length(nodes);
    edgecounter = 1;
    for source ∈ 1:n
        for target ∈ 1:n
            if haskey(edges, (source, target)) == true
                edgesinverse[edgecounter] = (source, target);
                edgecounter += 1;
            end
        end
    end
    # --------------------------------------------------------------------------- #

    # -- DO STUFF WITH CHILDREN ------------------------------------------------- #
    for id ∈ list_of_node_ids
        node = nodes[id];
        children[id] = _children(edges, node.id);
    end
    # --------------------------------------------------------------------------- #

    # -- DO STUFF WITH A -------------------------------------------------------- #
    d = length(edges);
    n = length(nodes);
    A = zeros(n, d);
    for (k,v) ∈ edgesinverse
        A[v[1], k] = -1.0;
        A[v[2], k] = 1.0;
    end
    # --------------------------------------------------------------------------- #
    
    # add stuff to model -
    graphmodel.nodes = nodes;
    graphmodel.edges = edges;
    graphmodel.edgesinverse = edgesinverse;
    graphmodel.children = children;
    graphmodel.A = A;

    # return -
    return graphmodel;
end
# --- PUBLIC METHODS ABOVE HERE ----------------------------------------------------------------------------------- #


