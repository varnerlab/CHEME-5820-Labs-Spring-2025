function learn(agent::MyDQNLearningAgentModel, worldmodel::Function; 
    context::MyDQNworldContextModel = nothing,
    maxnumberofsteps::Int = 192, 
    numberofepisodes::Int64 = 1, 
    γ::Float64 = 0.99,
    maxreplaybuffersize::Int64 = 1000, 
    trainfreq::Int64 = 10, 
    parameterupdatefreq::Int64 = 10,
    minibatchsize::Int64 = 64)::MyDQNLearningAgentModel

    # initialize -
    M = agent.mainnetwork; # main network
    T = agent.targetnetwork; # target network
    number_of_inputs = agent.number_of_inputs; # number of inputs
    actions = agent.actions; # actions
    K = length(actions); # number of actions that we have
    λ = 0.50; # learning rate (default: 0.01)
    β = 0.90; # momentum parameter (default: 0.90)
    loss(ŷ, y) = Flux.Losses.mse(ŷ, y, agg=mean); # loss function

    # setup the replay buffer -
    replaybuffer = CircularBuffer{Tuple{Vector{Float32},Vector{Float32}, Float32, Vector{Float32}}}(maxreplaybuffersize); # replay buffer

    # main loop -
    for i ∈ 1:numberofepisodes
        
        # stuff goes here -  
        optstate = Flux.setup(Momentum(λ, β), M); # we are optimize the parameters of the main network

        # time loop -
        s = zeros(Float32, number_of_inputs);
        s[1] = 1.0; # initial state
        for t ∈ 1:maxnumberofsteps
            
            # compute the ϵ -
            ϵₜ = (1.0/(t^(1/3)))*(log(K*t))^(1/3); # compute the epsilon value -
            
            # select an action -
            p = rand(); # role a random number
            aₜ = nothing;
            if p ≤ ϵₜ
                aₜ = rand(1:K) |> i-> actions[i]; # generate a random action
            elseif p > ϵₜ
                aₜ = M(s) |> U -> NNlib.softmax(U) |> p -> argmax(p) |> i -> actions[i]; # generate a greedy action
            end

            # implement action -
            s′, r = worldmodel(s, aₜ, context); # get the next state and reward for the world
            
            # store the transition in the replay buffer -
            push!(replaybuffer, (s, aₜ, r, s′)); # store the transition in the replay buffer
            if (isfull(replaybuffer) == true && rem(t, trainfreq) == 0)
                
                # sample a random batch from the replay buffer -
                minibatch = rand(replaybuffer, minibatchsize); # sample a random batch from the replay buffer
                
                # setup training minibatch (formulated in the proper format)
                for eⱼ ∈ minibatch
                    s̄, ā, r̄, n = eⱼ; # unpack the minibatch

                    # compute y (from the target network) -
                    y = r̄ + γ * (T(s̄) |> U -> argmax(p)) |> Float32 # compute the y value
                    
                    # compute the ŷ (from the main network) -
                    grads = Flux.gradient(M) do m
                        ŷ = m(n) |> U -> argmax(p) |> Float32 # compute the ŷ
                        loss(ŷ, y)
                    end

                    Flux.update!(optstate, M, grads[1])
                end
            end

            # check - update the target network?
            if (rem(t, parameterupdatefreq) == 0)
                T = deepcopy(M); # update the target network - this is a simple way to do it, but I'm assuming slow!
            end

            # do I need to do this?
            agent.mainnetwork = M; # update the main network
            agent.targetnetwork = T; # update the target network    
        end

        # reset -
        empty!(replaybuffer); # empty the replay buffers after each episode
    end

    return agent; # return the updated agent
end