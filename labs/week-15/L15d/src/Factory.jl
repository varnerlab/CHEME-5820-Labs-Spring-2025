

function MyCustomConvolutionLayerModel((nin, nout)::Pair, act=identity)
	
    W1 = Flux.glorot_uniform(nout, nin)
	W2 = Flux.glorot_uniform(nout, nin)
	b = fill!(similar(W1, nout), 0)
	
    return MyCustomConvolutionLayerModel(W1, W2, b, act)
end