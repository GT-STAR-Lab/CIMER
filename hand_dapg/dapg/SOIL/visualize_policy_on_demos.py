from statistics import mode
import mj_envs
import click 
import copy
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv
from traitlets.traitlets import default
import sys
DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy_on_demos --env_name relocate-v0 --policy policies/relocate-v0.pickle --demos demonstrations/relocate-v0_demos.pickle 
      --mode evaluation\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--config', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', required=True)
@click.option('--demos', type=str, help='absolute path of the demos file', required=False, default=None)
@click.option('--object_name', type=str, help='object for manipulation', default='')

def main(config, policy, demos, object_name):
    with open(config, 'r') as f:
        job_data = eval(f.read())
    task_id = job_data['env'].split('-')[0]
    if task_id == 'pen':
        task_horizon = 100
    elif task_id == 'relocate':
        task_horizon = 100
    elif task_id == 'door':
        task_horizon = 70
    elif task_id == 'hammer':
        task_horizon = 71  # crazy motions after hit
    else:
        print("Unkown task!")
        sys.exit()
    if task_id == 'relocate':
        Obj_sets = ['banana', 'cracker_box', 'cube', 'cylinder', 'foam_brick', 'gelatin_box', 'large_clamp', 'master_chef_can', 'mug', 'mustard_bottle', 'potted_meat_can', 'power_drill', 'pudding_box', 'small_ball', 'sugar_box', 'tomato_soup_can', 'tuna_fish_can']
        try: 
            if object_name not in Obj_sets:
                object_name = ''  # default setting
        except:
            object_name = '' # default setting
        e = GymEnv(job_data['env'], 'Torque', object_name)
    else:
        e = GymEnv(job_data['env'], 'Torque')
    pi = pickle.load(open(policy, 'rb'))
    if demos is not None:
        demos = pickle.load(open(demos, 'rb'))
    else:
        print("Please check the demo file.")
        sys.exit()
    # render policy
    e.visualize_policy_on_demos(pi, demos=demos, horizon=task_horizon, mode='evaluation', dapg_policy=False, Visualize=True, object_name=object_name)

if __name__ == '__main__':
    main()
