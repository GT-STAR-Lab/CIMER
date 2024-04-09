import mj_envs
import click 
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv
import requests

DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
def main(env_name):
    if env_name is "":
        print("Unknown env.")
        return
    demos = pickle.load(open('./demonstrations/'+env_name+'_demos.pickle', 'rb'))
    # render demonstrations
    demo_playback(env_name, demos)

def demo_playback(env_name, demo_paths):
    e = GymEnv(env_name)
    e.reset()
    is_first_state = True
    
    while True:
        actions = requests.get(url = 'http://localhost:5000/get').json()
        print(actions)
        if is_first_state:
            e.set_env_state(actions)
            is_first_state = False
        e.step(actions)
        e.env.mj_render()


if __name__ == '__main__':
    main()