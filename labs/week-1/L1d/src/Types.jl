abstract type MyAbstractUnsupervisedClusteringAlgorithm end

"""
    mutable struct MyNaiveKMeansClusteringAlgorithm

A mutable struct that represents a naive K-Means clustering algorithm. 
This struct is a subtype of `MyAbstractUnsupervisedClusteringAlgorithm`.

### Fields
- `K::Int64`: The number of clusters to create.
- `centroids::Dict{Int64, Vector{Float64}}`: The centroids of the clusters.
- `assignments::Vector{Int64}`: The cluster assignments.
- `ϵ::Float64`: The convergence threshold.
- `maxiter::Int64`: The maximum number of iterations.
- `dimension::Int64`: The dimension of the data.
- `number_of_points::Int64`: The number of data points.
"""
mutable struct MyNaiveKMeansClusteringAlgorithm <: MyAbstractUnsupervisedClusteringAlgorithm

    # data -
    K::Int64 # number of clusters
    centroids::Dict{Int64, Vector{Float64}} # cluster centroids
    assignments::Vector{Int64} # cluster assignments
    ϵ::Float64 # convergence criteria
    maxiter::Int64 # maximum number of iterations (alternatively, could use this convergence criterion)
    dimension::Int64 # dimension of the data
    number_of_points::Int64 # number of data points

    # constructor -
    MyNaiveKMeansClusteringAlgorithm() = new(); # build empty object
end