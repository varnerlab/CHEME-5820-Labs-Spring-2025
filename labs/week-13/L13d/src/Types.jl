abstract type AbstractTextRecordModel end
abstract type AbstractTextDocumentCorpusModel end
abstract type AbstractLayerModel end

# build a simple neural network model type -
struct MyFluxFeedForwardNeuralNetworkModel <: AbstractLayerModel
    chain::Chain; # holds the model chain
end

struct MyFluxElmanRecurrentNeuralNetworkModel <: AbstractLayerModel
    chain::Flux.Chain; # holds the model chain
end

struct MyFluxLSTMNeuralNetworkModel <: AbstractLayerModel
    chain::Flux.Chain; # holds the model chain
end

"""
    MySarcasmRecordModel <: AbstractTextRecordModel

### Fields 
- `data::Array{String, Any}`: The data found in the record in the order they were found
"""
mutable struct MySarcasmRecordModel <: AbstractTextRecordModel
    
    # data -
    issarcastic::Bool
    headline::String
    article::String
    
    # constructor -
    MySarcasmRecordModel() = new(); # empty
end

"""
    MySarcasmRecordCorpusModel <: AbstractTextDocumentCorpusModel

### Fields
- `records::Dict{Int, MySarcasmRecordModel}`: The records in the document (collection of records)
- `tokens::Dict{String, Int64}`: A dictionary of tokens in alphabetical order (key: token, value: position) for the entire document
"""
mutable struct MySarcasmRecordCorpusModel <: AbstractTextDocumentCorpusModel
    
    # data -
    records::Dict{Int, MySarcasmRecordModel}
    tokens::Dict{String, Int64}
    inverse::Dict{Int64, String}
    
    # constructor -
    MySarcasmRecordCorpusModel() = new(); # empty
end