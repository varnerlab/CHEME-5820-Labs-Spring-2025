# PRIVATE METHODS BELOW HERE ================================================================================= #
# placeholder - always return 0
_null(action::Int64)::Int64 = return 0;

function _solve(model::MyExploreFirstAlgorithmModel; T::Int = 0, world::Function = _null)::Array{Float64,2}

    # initialize -
    K = model.K # get the number of arms
    rewards = zeros(Float64, T, K); # rewards for each arm

    # how many expore steps should we take?
    Nₐ = ((T/K)^(2/3))*(log(T))^(1/3) |> x -> round(Int,x); # number of explore steps
    
    # exploration phase -
    counter = 1;
    for a ∈ 1:K
        for _ ∈ 1:Nₐ
            rewards[counter, a] = world(a); # store from action a
            counter += 1;
        end
    end

    μ = zeros(Float64, K); # average reward for each arm
    for a ∈ 1:K
        μ[a] = findall(x -> x != 0.0, rewards[:, a]) |> i-> mean(rewards[i, a]); # compute the average reward
    end

    # exploitation phase -
    a = argmax(μ); # compute the arm with best average reward
    for _ ∈ 1:(T - Nₐ*K)
        rewards[counter, a] = world(a); # store the reward
        counter += 1;
    end
    
    # return -
    return rewards;
end

function _solve(model::MyEpsilonGreedyAlgorithmModel; T::Int = 0, world::Function = _null)::Array{Float64,2}

    # initialize -
    K = model.K # get the number of arms
    rewards = zeros(Float64, T, K); # rewards for each arm

    for t ∈ 1:T
        ϵₜ = (1.0/(t^(1/3)))*(log(K*t))^(1/3); # compute the epsilon value -

        p = rand(); # role a random number
        aₜ = 1; # default action is to pull the first arm
        if (p ≤ ϵₜ)
            aₜ = rand(1:K);  # ramdomly select an arm
        else
            
            μ = zeros(Float64, K); # average reward for each arm
            for a ∈ 1:K
                μ[a] = findall(x -> x != 0.0, rewards[:, a]) |> i-> mean(rewards[i, a]); # compute the average reward
            end
            aₜ = argmax(μ); # compute the arm with best average reward
        end
        rewards[t, aₜ] = world(aₜ); # store the reward
    end

    # return -
    return rewards;
end

function _solve(model::MyUCB1AlgorithmModel; T::Int = 0, world::Function = _null)::Array{Float64,2}

    # initialize -
    K = model.K # get the number of arms
    rewards = zeros(Float64, T, K); # rewards for each arm
    Nₐ = zeros(Int64, K); # number of times we have pulled each arm

    # try each arm once
    counter = 1;
    for a = 1:K
        rewards[counter, a] = world(a); # pull each arm once
        Nₐ[a] += 1; # increment the counter
        counter += 1;
    end
    
    # main loop -
    for t ∈ counter:T

        # conpute the UCB value 
        tmp = zeros(Float64, K);
        μ = zeros(Float64, K); # average reward for each arm
        for a ∈ 1:K
            μ[a] = findall(x -> x != 0.0, rewards[:, a]) |> i-> mean(rewards[i, a]); # compute the average reward
        end
        
        for i ∈ 1:K
            tmp[i] = μ[i] + sqrt((2*log(t))/Nₐ[i]); # compute the UCB value
        end

        aₜ = argmax(tmp); # select the arm with the highest UCB value
        Nₐ[aₜ] += 1; # increment the counter
        rewards[t, aₜ] = world(aₜ); # store the reward
    end

    # return -
    return rewards;
end


# PRIVATE METHODS ABOVE HERE ================================================================================= #

# PUBLIC METHODS BELOW HERE ================================================================================== #`
function solve(model::AbstractBanditAlgorithmModel; T::Int = 0, world::Function = _null)
    return _solve(model, T = T, world = world);
end
# PUBLIC METHODS ABOVE HERE ================================================================================== #`