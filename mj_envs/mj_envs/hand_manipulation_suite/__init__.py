from gym.envs.registration import register
from mjrl.envs.mujoco_env import MujocoEnv

register(
    id='door-allegro-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorAllegroEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.door_allegro_v0 import DoorAllegroEnvV0

# register -> to register the environments used in the experiments -> specify the system dynamics
# Swing the door open
register(
    id='door-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0

# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
)
from mj_envs.hand_manipulation_suite.pen_v0 import PenEnvV0

# Relcoate an object to the target
register(
    id='relocate-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

# Relcoate an object to the target (original dapg env)
register(
    id='relocate-v1',
    entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV1',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.relocate_v1 import RelocateEnvV1
