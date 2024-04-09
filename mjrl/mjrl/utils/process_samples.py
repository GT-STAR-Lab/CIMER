import numpy as np

# rollout method to compute the (return) Q value -> large variance and small bias. Using GAE mode for advantage function, we expect to reduce variance at the cose of some bias 
# by introducing an approximate value function
def compute_returns(paths, gamma): # Q value function -> the sum of rewards after implementing this action, 
    # because the rewards were obtained after the action being executed -> take consideration of the effect of selected action -> Q value
    for path in paths: # each path is a dict
        path["returns"] = discount_sum(path["rewards"], gamma) # obtained the rewards and add to the dict

def compute_advantages(paths, baseline, gamma, gae_lambda=None, normalize=False):
    # compute and store returns, advantages, and baseline 
    # standard mode
    # Advantage function is nothing but difference between Q value(discounted over time) for a given state — action pair and value function of the state.
    # However, impletation-wise, Q and V values are only approximated using the generated samples -> we don't know the exact functions of Q,V

    # this method is advantage learning (compared with GAE method), that is using the Q(Monte-Carlo sampled reward signal) and V(s) (parameterized value estimate using the baseline model)
    # A = Q(returns) - V (with high-variance, since Q is obtained using rollout) but Q take into the effect of the current action.
    # the oldest version of computing the advantage
    if gae_lambda == None or gae_lambda < 0.0 or gae_lambda > 1.0:
        for path in paths:
            path["baseline"] = baseline.predict(path) # baseline model is being considered as an approximate value function value^{\pi}
            path["advantages"] = path["returns"] - path["baseline"]  # path["returns"] -> the reward of the whole traj using the current policy
        if normalize:
            alladv = np.concatenate([path["advantages"] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path["advantages"] = (path["advantages"]-mean_adv)/(std_adv+1e-8)
    # GAE mode, as pointed out in the paper (Generalized Advantage Estimate)
    # paper: HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION
    # the modern way to compute the advantage
    # By setting gae_lambda to 0, the algorithm reduces to TD learning (high bias, mainly using the estimated value function V(s) with large bias), while setting it to 1 produces Monte-Carlo sampling (high variance)
    # gae_lambda in-between (particularly those in the 0.9 to 0.999 range) produce better empirical performance by trading off the bias of V(s) with the variance of the trajectory.\
    # V value function in the paper represents the baseline model
    else: 
        for path in paths:
            b = path["baseline"] = baseline.predict(path) # approximate the value function using another neural network
            # "baseline.predict(path)" is used to compute the V values of the currently generated trajectories. 
            if b.ndim == 1:
                b1 = np.append(path["baseline"], 0.0 if path["terminated"] else b[-1])
            else:
                b1 = np.vstack((b, np.zeros(b.shape[1]) if path["terminated"] else b[-1]))
            td_deltas = path["rewards"] + gamma*b1[1:] - b1[:-1]  # delta value at each time step (TD residual term), b1 -> appriximate value functions V from the baseline model
            # the baseline momdel always has to be optimized during each iteration
            # these are exactly the same in the paper
            path["advantages"] = discount_sum(td_deltas, gamma*gae_lambda)  # gamma and lambda, as named in the paper (to compute the advantage values)
            # GAE advantage is the gamma*gae_lambda-weighted average of these delta values
        if normalize:
            alladv = np.concatenate([path["advantages"] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path["advantages"] = (path["advantages"]-mean_adv)/(std_adv+1e-8)

def discount_sum(x, gamma, terminal=0.0):
    y = []
    run_sum = terminal  
    # print("len(x): ", len(x))   # task horizon 
    for t in range(len(x)-1, -1, -1):  # starting from the last one (reward should be very large since the task is almost finished)
    # at early stage when t is smaller, the rewards are small (<0, can be seen as penelty).
        run_sum = x[t] + gamma*run_sum # cumulative rewards for the whole traj (gamma = 0.995), and there is a reward value for each state
        y.append(run_sum)
    return np.array(y[::-1])