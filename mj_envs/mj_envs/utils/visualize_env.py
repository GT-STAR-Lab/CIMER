import gym
import mj_envs
import click 
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python visualize_policy.py --env_name door-v0 \n
    $ python visualize_policy.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('--episodes', type=int, help='number of episodes to visualize', default=10)
@click.option('--time', type=int, help='length for simulation', default=100)
def main(env_name, policy, mode, seed, episodes, time):
    e = GymEnv(env_name, "PID")
    e.set_seed(seed)

    # NOTE: mod by CZY
    obs_dim = e.observation_dim
    ac_dim  = e.action_dim
    print("environment setup is: ", e.spec)
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        # pi = MLP(e.spec, hidden_sizes=(32,32), seed=seed, init_log_std=-1.0)
        pi = MLP(obs_dim, ac_dim, hidden_sizes=(32,32), seed=seed, init_log_std=-1.0) # random generated network
        
    # render policy
    # ori, init_pos, init_vel = 0,0,0
    # e.visualize_policy(pi, "test_policy", ori, init_pos, init_vel, num_episodes=episodes, horizon=e.horizon, mode=mode, record=False)
    for t in range(time):
        e.render()

if __name__ == '__main__':
    main()
