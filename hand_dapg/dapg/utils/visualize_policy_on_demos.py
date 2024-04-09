import mj_envs
import click 
import copy
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv
from traitlets.traitlets import default

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy_on_demos --env_name relocate-v0 --policy policies/relocate-v0.pickle --demos demonstrations/relocate-v0_demos.pickle 
      --mode evaluation\n
'''

init_state_dict_per_env = {
    'relocate-v0': {
        'qpos': np.zeros((36,), dtype=np.float64),
        'qvel': np.zeros((36,), dtype=np.float64),
        'hand_qpos': np.zeros((30,), dtype=np.float64),
        'palm_pos': np.array([-0.00692036, -0.19996033, 0.15038709]),
        # 'obj_pos': np.array([-0.33460906, 0.14691826, 0.035]),
        'obj_pos': np.array([-0.3, 0.3, 0.035]),
        # 'target_pos': np.array([ -0.08026819, -0.03606687, 0.25210981])
        'target_pos': np.array([ 0.3, -0.3, 0.25210981])
        # 'target_pos': np.array([ -0.13026819, -0.03606687, 0.25210981])
    }
}

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', required=True)
@click.option('--demos', type=str, help='absolute path of the demos file', required=False, default=None)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--object_name', type=str, help='object for manipulation', default=None)
@click.option('--visualize', type=str, help='object for manipulation', required=True, default=None)
def main(env_name, policy, demos, mode, object_name, visualize):
    Visualize = True if visualize == 'True' else False 
    if env_name == 'relocate-v0':
        Obj_sets = ['banana', 'cracker_box', 'cube', 'cylinder', 'foam_brick', 'gelatin_box', 'large_clamp', 'master_chef_can', 'mug', 'mustard_bottle', 'potted_meat_can', 'power_drill', 'pudding_box', 'small_ball', 'sugar_box', 'tomato_soup_can', 'tuna_fish_can']
        try: 
            if object_name not in Obj_sets:
                object_name = ''  # default setting
        except:
            object_name = '' # default setting
        e = GymEnv(env_name, 'Torque', object_name)
    else:
        e = GymEnv(env_name, 'Torque')
    pi = pickle.load(open(policy, 'rb'))
    if demos is not None:
        demos = pickle.load(open(demos, 'rb'))
    elif env_name in init_state_dict_per_env:
        demos = [{
            'init_state_dict': copy.deepcopy(init_state_dict_per_env[env_name])
        }]
    # render policy
    e.visualize_policy_on_demos(pi, demos=demos, horizon=100, mode=mode, Visualize=Visualize, object_name=object_name)

if __name__ == '__main__':
    main()
