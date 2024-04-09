import requests
from mjrl.utils.gym_env import GymEnv

e = GymEnv('relocate-v0')
e.reset()
is_first_state = True

while True:
    actions = requests.get(url = 'http://localhost:5000/get')
    print(actions)
    if is_first_state:
        e.set_env_state(actions)
        is_first_state = False
    e.step(actions)
    e.env.mj_render()

