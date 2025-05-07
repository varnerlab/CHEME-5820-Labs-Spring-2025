"""
    learn(agent::MyDQNLearningAgentModel, worldmodel::Function; 
        context::MyDQNworldContextModel = nothing,
        maxnumberofsteps::Int = 192, 
        numberofepisodes::Int64 = 1, 
        γ::Float64 = 0.95,
        maxreplaybuffersize::Int64 = 1000, 
        trainfreq::Int64 = 10, 
        parameterupdatefreq::Int64 = 10,
        minibatchsize::Int64 = 64) -> MyDQNLearningAgentModel

A method to learn a DQN agent using a world model.

### Arguments
    - agent::MyDQNLearningAgentModel: the DQN agent to learn
    - worldmodel::Function: the world model to use
    - context::MyDQNworldContextModel: the context of the world model
    - maxnumberofsteps::Int: the maximum number of steps to take in each episode. Default is 192.
    - numberofepisodes::Int64: the number of episodes to run. Default is 1.
    - γ::Float64: the discount factor. Default is 0.95.
    - maxreplaybuffersize::Int64: the maximum size of the replay buffer. Default is 1000.
    - trainfreq::Int64: the frequency of training. Default is 10.
    - parameterupdatefreq::Int64: the frequency of parameter updates. Default
    - minibatchsize::Int64: the size of the minibatch. Default is 64.

### Returns
    - agent::MyDQNLearningAgentModel: the learned DQN agent with updated parameters

"""
function learn(agent::MyDQNLearningAgentModel, worldmodel::Function; 
    context::MyDQNworldContextModel = nothing,
    maxnumberofsteps::Int = 192, 
    numberofepisodes::Int64 = 1, 
    γ::Float64 = 0.95,
    maxreplaybuffersize::Int64 = 1000, 
    trainfreq::Int64 = 10, 
    parameterupdatefreq::Int64 = 10,
    minibatchsize::Int64 = 64)::MyDQNLearningAgentModel

    # initialize -
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
        
        # Get the networks, and setup the optimizer - 
        M = agent.mainnetwork; # main network
        T = agent.targetnetwork; # target network
        optstate = Flux.setup(Descent(), M); # we are optimize the parameters of the main network
        
        println("Episode: ", i); # print the episode number

        # time loop -
        s = ones(Float32, number_of_inputs);
        for t ∈ 1:maxnumberofsteps
            
            # compute the ϵ -
            ϵₜ = (1.0/(t^(1/3)))*(log(K*t))^(1/3); # compute the epsilon value -
            
            # select an action -
            p = rand(); # role a random number
            aₜ = nothing;
            if p ≤ ϵₜ
                aₜ = rand(1:K) |> i-> actions[i]; # generate a random action
            elseif p > ϵₜ
                aₜ = M(s) |> U -> argmax(U) |> i -> actions[i]; # generate a greedy action
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
                    y = r̄ + γ * (T(n) |> U -> argmax(U)) |> Float32 # compute the y value
                    
                    # compute the ŷ (from the main network) -
                    grads = Flux.gradient(M) do m
                        ŷ = m(s̄) |> U -> argmax(U) |> Float32 # compute the ŷ
                        loss(ŷ, y)
                    end

                    Flux.update!(optstate, M, grads[1])
                end
            end

            # check - update the target network?
            if (rem(t, parameterupdatefreq) == 0)
                modelstate = Flux.state(M); # update the target network
                Flux.loadmodel!(T, modelstate);
            end

            # do I need to do this?
            agent.mainnetwork = M; # update the main network
            agent.targetnetwork = T; # update the target network 
            
            # update the state -
            s = s′; # update the state
        end

        # reset -
        # empty!(replaybuffer); # empty the replay buffers after each episode
    end

    # grab the replay buffer, store in agent -
    agent.replaybuffer = replaybuffer; # store the replay buffer in the agent

    return agent; # return the updated agent
end