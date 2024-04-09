"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp_soil import MLP
from mjrl.baselines.mlp_baseline_soil import MLPBaseline
from mjrl.algos.soil import SOIL
from mjrl.utils.train_agent_soil import train_agent
import os
import json
import mjrl.envs
import mj_envs   # read the env files (task files)
import time as timer
import pickle
import argparse

# ===============================================================================
# Get command line arguments
# ===============================================================================

class Spec:
    def __init__(self, env=None, env_name="relocate-mug-1"):
        self.observation_dim = e.spec.observation_dim
        self.action_dim = e.spec.action_dim
        self.env_id = env_name

parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data.')
parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
parser.add_argument('--eval_data', type=str, required=True, help='absolute path to evaluation data')
args = parser.parse_args()
JOB_DIR = args.output
if not os.path.exists(JOB_DIR):
    os.mkdir(JOB_DIR)
with open(args.config, 'r') as f:
    job_data = eval(f.read())
assert 'algorithm' in job_data.keys()
assert any([job_data['algorithm'] == a for a in ['SOIL', 'RL']])
assert any([job_data['pg_algo'] == a for a in ['npg', 'trpo']])
job_data['lam_0'] = 0.0 if 'lam_0' not in job_data.keys() else job_data['lam_0']
job_data['lam_1'] = 0.0 if 'lam_1' not in job_data.keys() else job_data['lam_1']
EXP_FILE = JOB_DIR + '/job_config.json'
with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)
task_id = job_data['env'].split('-')[0]
if task_id == 'pen':
    task_horizon = 100
elif task_id == 'relocate':
    task_horizon = 100
elif task_id == 'door':
    task_horizon = 70
elif task_id == 'hammer':
    task_horizon = 31
# ===============================================================================
# Train Loop
# ===============================================================================

e = GymEnv(job_data['env'], 'Torque')
policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['fixed_seed'], horizon=task_horizon)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                       epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])  # the baseline model used
# Get demonstration data if necessary and behavior clone (only not NPG?)
demo_paths = pickle.load(open(args.eval_data, 'rb'))
if job_data['demo_traj'] > 0 and job_data['demo_traj'] < len(demo_paths):
    demo_paths = demo_paths[:job_data['demo_traj']]

# ===============================================================================
# RL Loop
# ===============================================================================
spec = Spec(e, e.env_id)
soil_param = {'SOIL_iter': job_data['SOIL_iter'], 'SOIL_MB_SIZE': job_data['SOIL_MB_SIZE'], 'SOIL_LR': job_data['SOIL_LR'], 'SOIL_WD': job_data['SOIL_WD'], 'SOIL_RBS': job_data['SOIL_RBS'], 'SOIL_ADV_W': job_data['SOIL_ADV_W'], 'SOIL_MLP_W': job_data['SOIL_MLP_W']}
# So we always use SOIL for the training, no 
rl_agent = SOIL(
    spec, e, policy, baseline,
    demo_paths=demo_paths,
    alg_name=job_data['algorithm'],
    normalized_step_size=0.1, seed=job_data['RL_seed'],
    lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
    save_logs=True,soil_param = soil_param,pg_algo=job_data['pg_algo']
) # step size is fixed to be 0.05 (0.1 / 2)
print("========================================")
print("Starting reinforcement learning phase")
print("========================================")

ts = timer.time()
train_agent(job_name=JOB_DIR,
            agent=rl_agent,
            seed=job_data['fixed_seed'],
            niter=job_data['rl_num_iter'], 
            gamma=job_data['rl_gamma'],  # 
            gae_lambda=job_data['rl_gae'], # 
            num_cpu=job_data['num_cpu'],
            sample_mode='trajectories',  # train_mode = trajectories
            num_traj=job_data['rl_num_traj'],
            save_freq=job_data['save_freq'],
            evaluation_rollouts=job_data['eval_rollouts'])
print("time taken = %f" % (timer.time()-ts))

print("Training finishes.")