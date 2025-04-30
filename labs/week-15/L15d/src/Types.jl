

struct MyCustomConvolutionLayerModel{A<:AbstractMatrix, B<:AbstractVector, F} <: GNNLayer
	W1::A
	W2::A
	b::B
	act::F
end