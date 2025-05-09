
function eval_loss_accuracy(model, data_loader, device)
    loss = 0.
    acc = 0.
    ntot = 0
    for (g, y) in data_loader
        g, y = g |> device, y |> device
        n = length(y)
        ŷ = model(g, g.ndata.x)
        loss += logitcrossentropy(ŷ, y) * n 
        acc += mean((ŷ .> 0) .== y) * n
        ntot += n
    end 
    return (loss = round(loss/ntot, digits=4), acc = round(acc*100/ntot, digits=2))
end

function train!(model; epochs=200, η=1e-2, infotime=10)
	
    # device = Flux.gpu # uncomment this for GPU training
	device = Flux.cpu
	model = model |> device
	opt_state = Flux.setup(Adam(η), model)
	

    function report(epoch)
        train = eval_loss_accuracy(model, train_loader, device)
        test = eval_loss_accuracy(model, test_loader, device)
        println("# epoch = $epoch")
        println("train = $train")
        println("test = $test")
    end
    
    report(0)
    for epoch in 1:epochs
        for (g, y) in train_loader
            g, y = g |> device, y |> device
            grads = Flux.gradient(model) do model
                ŷ = model(g, g.ndata.x)
                logitcrossentropy(ŷ, y)
            end
            Flux.Optimise.update!(opt_state, model, grads[1])
        end
		epoch % infotime == 0 && report(epoch)
    end
end


function (l::MyCustomConvolutionLayerModel)(g::GNNGraph, x::AbstractMatrix)
	
    message(xi, xj, e) = l.W2 * xj
	m = apply_edges(message, g, xj=x) # size [nout, num_edges]
	xnew = aggregate_neighbors(g, +, m) # size [nout, num_nodes]
	# m = propagate(message, g, +, xj=x) # equivalent to the 2 lines above
	return l.act.(l.W1*x .+ xnew .+ l.b)
end