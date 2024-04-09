import mj_envs
import click 
import os
import gym
import numpy as np
import pickle
from scipy.spatial.transform import Rotation
from mjrl.utils.gym_env import GymEnv
from scipy.spatial.transform import Rotation as R

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy --env_name relocate-v0 --policy policies/relocate-v0.pickle --mode evaluation\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', required=True)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
# @click.option('--record', help='record the demonstrations', is_flag=True)
def main(env_name, policy, mode):
    record = False
    e = GymEnv(env_name) # e.horizon = 100
    print("horizon:", e.horizon)
    pi = pickle.load(open(policy, 'rb'))
    # render policy
    orientation = []
    desired_ori = np.zeros(3)
    # desired_ori[0] -> pitch, desired_ori[1] -> yaw, desired_ori[2] -> roll
    desired_ori[0] = np.pi / 4 # pitch 
    desired_ori[1] = 0 # pitch 
    # desired_ori[1] = np.pi / 4  # yaw
    desired_ori[2] = 0 # roll does not matter
    orientation.append(desired_ori)
    desired_ori = np.zeros(3)
    desired_ori[0] = 0 # pitch 
    desired_ori[1] = 0 # pitch 
    # desired_ori[1] = 0 # pitch 
    orientation.append(desired_ori)
    desired_ori = np.zeros(3)
    desired_ori[0] = -np.pi / 4 # pitch 
    desired_ori[1] = 0 # pitch 
    # desired_ori[1] = -np.pi / 4 # pitch 
    orientation.append(desired_ori)
    # desired_ori[0] = 0.8
    # desired_ori[1] = -0.8
    # desired_ori[2] = 0
    init_position_range = 0.10  # RNN:0.15 (100%)
    init_velocity_range = 0.10
    # num_episodes -> num of demo traj
    # horizon -> task length
    # e.horizon = 100
    e.visualize_policy(pi, policy_name="Pen_task_1000.pickle", ori=orientation, init_pos=init_position_range, init_vel=init_velocity_range,num_episodes=1500, horizon=e.horizon, mode=mode, record=record)

if __name__ == '__main__':
    main()