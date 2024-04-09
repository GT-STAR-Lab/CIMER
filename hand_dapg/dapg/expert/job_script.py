"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp_soil import MLP
from mjrl.baselines.mlp_baseline_soil import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning import BC
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

parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data.')
parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
parser.add_argument('--onlybc', help='whether or not do only BC', type=str, default=False)
parser.add_argument('--pre_trained_path', type=str, required=False, default='none', help='if not none -> training on new objects for testing quick adaptation')
args = parser.parse_args()
is_only_bc = True if args.onlybc == 'True' else False
JOB_DIR = args.output
if not os.path.exists(JOB_DIR):
    os.mkdir(JOB_DIR)
with open(args.config, 'r') as f:
    job_data = eval(f.read())
assert 'algorithm' in job_data.keys()
assert any([job_data['algorithm'] == a for a in ['NPG', 'BCRL', 'DAPG']])
# NPG -> no demo data, rl learn from scratch (no bc init)
# BCRL -> BC init, then NPG fine tune
# DAPG -> BC init, then DAPG fine tune (with augmented loss function)
# when using BCRL or DPAG, make sure that there is always a demo path being provided in txt file.
job_data['lam_0'] = 0.0 if 'lam_0' not in job_data.keys() else job_data['lam_0']
job_data['lam_1'] = 0.0 if 'lam_1' not in job_data.keys() else job_data['lam_1']
EXP_FILE = JOB_DIR + '/job_config.json'
with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)

# ===============================================================================
# Train Loop
# ===============================================================================    
if job_data['env'] == 'relocate-v1':
    Obj_sets = ['banana', 'cracker_box', 'cube', 'cylinder', 'foam_brick', 'gelatin_box', 'large_clamp', 'master_chef_can', 'mug', 'mustard_bottle', 'potted_meat_can', 'power_drill', 'pudding_box', 'small_ball', 'sugar_box', 'tomato_soup_can', 'tuna_fish_can']
    try: 
        if job_data['object_name'] not in Obj_sets:
            object_name = ''  # default setting
        else:
            object_name = job_data['object_name']
    except:
        object_name = '' # default setting
    e = GymEnv(job_data['env'], 'Torque', object_name)
else:
    e = GymEnv(job_data['env'], 'Torque')
print("observation dimension: ", e.spec.observation_dim)
print("action dimension: ", e.spec.action_dim)
print("horizon: ", e.spec.horizon)
print("EnvID: ", e.env_id)
print("action dimension: ", e.horizon) 
task_id = job_data['env'].split('-')[0]
if task_id == 'pen':
    task_horizon = 100
elif task_id == 'relocate':
    task_horizon = 100
elif task_id == 'door':
    task_horizon = 70
elif task_id == 'hammer':
    task_horizon = 31
policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['fixed_seed'], horizon=task_horizon)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                       epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])  # the baseline model used

# Get demonstration data if necessary and behavior clone (only not NPG?)
demo_paths = None
if job_data['algorithm'] != 'NPG': # != 'NPG' -> no demo data, NPG train from scratch (make sure --onlybc sets to be False)
    print("========================================")
    print("Collecting expert demonstrations")
    print("========================================")
    demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))   # list object, number of demos
    
    # 25 of demo paths
    # print(demo_paths[1].keys())
    # print(demo_paths[1]['init_state_dict'])
    if args.pre_trained_path == 'none':  # training from scratch, instead of applying fine-tuning on new objects
        bc_agent = BC(demo_paths, policy=policy, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                    lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)  # pre-training on the model policy
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running BC with expert demonstrations")
        print("========================================")
        bc_agent.train(save_policy=is_only_bc, task=job_data['env'].split('-')[0])
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")
    else:  # loading the policies that are pre-trained on other objects and quick adapt on new objects
        print("load the policy from %s"%(args.pre_trained_path))
        with open(args.pre_trained_path, 'rb') as fp:
            policy = pickle.load(fp)
        baseline_path = (args.pre_trained_path[::-1].replace('policy'[::-1],'baseline'[::-1], 1))[::-1]
        pickle.dump(policy, open(JOB_DIR + '/Pre_trained_policy.pickle', 'wb'))
        with open(baseline_path, 'rb') as fp:
            baseline = pickle.load(fp)

if not is_only_bc:  # NPG or BCRL or Dapg
    if job_data['algorithm'] != 'DAPG':  # bcrl, will not use the demo data during RL training -> demo date is only used for warm start
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    # ===============================================================================
    # RL Loop
    # ===============================================================================

    # So we always use DAPG for the training, no 
    rl_agent = DAPG(e, policy, baseline, demo_paths,  # damo_path loads the pickle file
                    normalized_step_size=job_data['rl_step_size'],
                    lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
                    seed=job_data['RL_seed'], save_logs=True
                    )
    print("========================================")
    print("Starting reinforcement learning phase")  
    print("========================================")

    ts = timer.time()
    train_agent(job_name=JOB_DIR,
                agent=rl_agent,
                e=e,
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