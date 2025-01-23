"""
    build(modeltype::Type{MyNaiveKMeansClusteringAlgorithm}, data::NamedTuple)::MyNaiveKMeansClusteringAlgorithm

Build a naive k-means clustering model.

### Arguments
- `modeltype::Type{MyNaiveKMeansClusteringAlgorithm}`: The type of the model to build.
- `data::NamedTuple`: The data to use to build the model.

The `data::NamedTuple` must contain the following fields:
- `K::Int64`: The number of clusters to create.
- `ϵ::Float64`: The convergence threshold.
- `maxiter::Int64`: The maximum number of iterations.
- `dimension::Int64`: The dimension of the data.
- `number_of_points::Int64`: The number of data points.
- `scale_factor::Float64`: The scale factor to use when generating the initial centroids.

### Returns
- `model::MyNaiveKMeansClusteringAlgorithm`: The naive k-means clustering model with populated (initial) centroids and assignments.
"""
function build(modeltype::Type{MyNaiveKMeansClusteringAlgorithm}, data::NamedTuple)::MyNaiveKMeansClusteringAlgorithm
    
    # build an empty model -
    model = modeltype();

    # get data -
    K = data.K;
    ϵ = data.ϵ;
    maxiter = data.maxiter;
    dimension = data.dimension;
    number_of_points = data.number_of_points;
    SF = data.scale_factor;

    # setup the initial assignments -
    assignments = zeros(Int64, number_of_points);
    for i ∈ 1:number_of_points
        assignments[i] = rand(1:K); # randomly assign points to clusters
    end

    # setup the centriods -
    centroids = Dict{Int64, Vector{Float64}}();
    for k ∈ 1:K
        centroids[k] = SF*rand(Float64, dimension); # randomly generate the centriods
    end

    # set the data on the model -
    model.K = K;
    model.ϵ = ϵ;
    model.maxiter = maxiter;
    model.dimension = dimension;
    model.number_of_points = number_of_points;
    model.assignments = assignments;
    model.centroids = centroids;

    # return the model -
    return model;
end