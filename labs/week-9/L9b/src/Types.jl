abstract type AbstractlHopfieldNetworkModel end


mutable struct MyClassicalHopfieldNetworkModel <: AbstractlHopfieldNetworkModel

    # data -
    W::Array{<:Number, 2} # weight matrix
    b::Array{<:Number, 1} # bias vector
    energy::Dict{Int64, Float32} # energy of the states

    # empty constructor -
    MyClassicalHopfieldNetworkModel() = new();
end