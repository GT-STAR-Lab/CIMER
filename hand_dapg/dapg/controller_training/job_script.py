"""
This is a job script for controller learning for KODex 1.0
"""
import sys
sys.path.append('/home/pshah479/Desktop/mjrl_repo/CIMER/mjrl')


from re import A
from mjrl.utils.resnet import *
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.ppo_clip import PPO
from mjrl.algos.behavior_cloning_KODex import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
from mjrl.KODex_utils.Observables import *
from mjrl.KODex_utils.quatmath import quat2euler, euler2quat
from mjrl.KODex_utils.coord_trans import ori_transform, ori_transform_inverse
from mjrl.KODex_utils.Controller import *
import sys
import os
import json
import mjrl.envs
import mj_envs   # read the env files (task files)
import time as timer
import pickle
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ",device)
def demo_playback(e, resnet_model, demo_paths, num_demo, task_id):
    Training_data = []
    print("Begin loading demo data!")
    # sample_index = np.random.choice(len(demo_paths), num_demo, replace=False)  # Random data is used
    sample_index = range(num_demo)
    for t in sample_index:  # only read the initial conditions
        path = demo_paths[t]
        state_dict = {}
        if task_id == 'pen':
            observations = path['observations']  
            handVelocity = path['handVelocity']  
            obs = observations[0] # indeed this one is defined in the world frame(fixed on the table) (for object position and object orientations)
            state_dict['init_states'] = path['init_state_dict']
            state_dict['handpos'] = obs[:24]
            state_dict['handvel'] = handVelocity[0]
            state_dict['objpos'] = obs[24:27] # in the world frame(on the table)
            state_dict['objvel'] = obs[27:33]
            state_dict['desired_ori'] = obs[36:39] # desired orientation (an unit vector in the world frame)
            state_dict['objorient'] = obs[33:36] # initial orientation (an unit vector in the world frame)
        elif task_id == 'relocate':
            e.reset()
            e.set_env_state(path['init_state_dict'])
            observations = path['observations']  
            observations_visualize = path['observations_visualization']
            handVelocity = path['handVelocity'] 
            obs = observations[0] 
            obs_visual = observations_visualize[0]
            rgb, depth = e.env.mj_render()
            # plt.imshow(e.env.mj_render()[0])
            # plt.savefig("/home/pratik/Desktop/mjrl_repo/CIMER_KOROL/CIMER/hand_dapg/dapg/controller_training/outputttt.jpg")
            # break
            rgb = (rgb.astype(np.uint8) - 128.0) / 128
            depth = depth[...,np.newaxis]
            rgbd = np.concatenate((rgb,depth),axis=2)
            rgbd = np.transpose(rgbd, (2, 0, 1))
            rgbd = rgbd[np.newaxis, ...]
            rgbd = torch.from_numpy(rgbd).float().to(device)
            # desired_pos = Test_data[k][0]['init']['target_pos']
            desired_pos=obs[45:48]

            desired_pos = desired_pos[np.newaxis, ...]
            desired_pos = torch.from_numpy(desired_pos).float().to(device)
            implict_objpos = resnet_model(rgbd, desired_pos) 
            obj_OriState = implict_objpos[0].cpu().detach().numpy()
            
            state_dict['init_states'] = path['init_state_dict']
            state_dict['handpos'] = obs[:30]
            state_dict['handvel'] = handVelocity[0][:30]
            objpos = obs[39:42] # in the world frame(on the table)
            state_dict['desired_pos'] = obs[45:48] 
            state_dict['objpos'] = objpos - obs[45:48] # converged object position
            state_dict['objorient'] = obs_visual[33:36]
            state_dict['objvel'] = handVelocity[0][30:]
            state_dict['obj_features']=obj_OriState
        elif task_id == 'door':
            observations = path['observations']  
            observations_visualize = path['observations_visualization']
            obs = observations[0] 
            obs_visual = observations_visualize[0]
            state_dict['init_states'] = path['init_state_dict']
            state_dict['handpos'] = obs_visual[:28]
            state_dict['handvel'] = obs_visual[30:58]
            state_dict['objpos'] = obs[32:35]
            state_dict['objvel'] = obs_visual[58:59]
            state_dict['handle_init'] = path['init_state_dict']['door_body_pos']
        elif task_id == 'hammer':
            observations = path['observations'] 
            handVelocity = path['handVelocity'] 
            obs = observations[0]
            allvel = handVelocity[0]
            state_dict['handpos'] = obs[:26]
            state_dict['handvel'] = allvel[:26]
            state_dict['objpos'] = obs[49:52] + obs[42:45] 
            state_dict['objorient'] = obs[39:42]
            state_dict['objvel'] = obs[27:33]
            state_dict['nail_goal'] = obs[46:49]
        Training_data.append(state_dict)
    print("Finish loading demo data!")
    return Training_data



#Resnet Model


resnet_model = ClassificationNetwork18_woDCT(feat_dim = 8)
resnet_model = resnet_model.float()
resnet_model.eval()
resnet_model = resnet_model.to(device)
# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data.')
parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
parser.add_argument('--eval_data', type=str, required=True, help='absolute path to evaluation data')
parser.add_argument('--resnet_weights',type=str,required=True, help='absolute path to resnet model weights' )
parser.add_argument('--pre_trained_path', type=str, required=False, default='none', help='if not none -> training on new objects for testing quick adaptation')
args = parser.parse_args()
with open(args.config, 'r') as f:
    job_data = eval(f.read())
assert 'algorithm' in job_data.keys()
assert any([job_data['algorithm'] == a for a in ['NPG', 'TRPO', 'PPO']])  # start from natural policy gradient for training
assert os.path.exists(os.getcwd() + job_data['matrix_file'] + job_data['env'].split('-')[0] + '/koopmanMatrix.npy') 
JOB_DIR = args.output
if not os.path.exists(JOB_DIR):
    os.mkdir(JOB_DIR)
EXP_FILE = JOB_DIR + '/job_config.json'
with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)
KODex = np.load(os.getcwd() + job_data['matrix_file'] + job_data['env'].split('-')[0] + '/koopmanMatrix.npy')  # loading KODex reference dynamics
print(KODex)
# ===============================================================================
# Set up the controller parameter
# ===============================================================================
# This set is used for control frequency: 500HZ (for all DoF)
PID_P = 10
PID_D = 0.005  
Simple_PID = PID(PID_P, 0.0, PID_D)

# loading model
PATH=args.resnet_weights
resnet_model.load_state_dict(torch.load(PATH, weights_only=True))
resnet_model.eval()
for param in resnet_model.parameters():
  print(param.data)
  break
# ===============================================================================
# Task specification
# ===============================================================================
task_id = job_data['env'].split('-')[0]
if task_id == 'pen':
    num_robot_s = 24
    num_object_s = 12
    task_horizon = 100
elif task_id == 'relocate':
    Obj_sets = ['banana', 'cracker_box', 'cube', 'cylinder', 'foam_brick', 'gelatin_box', 'large_clamp', 'master_chef_can', 'mug', 'mustard_bottle', 'potted_meat_can', 'power_drill', 'pudding_box', 'small_ball', 'sugar_box', 'tomato_soup_can', 'tuna_fish_can']
    try: 
        if job_data['object'] not in Obj_sets:
            job_data['object'] = ''  # default setting
    except:
        job_data['object'] = '' # default setting
    num_robot_s = 30
    num_object_s = 8
    task_horizon = 100
elif task_id == 'door':
    num_robot_s = 28
    num_object_s = 7
    task_horizon = 70
elif task_id == 'hammer':
    num_robot_s = 26
    num_object_s = 15
    task_horizon = 31
else:
    print("Unkown task!")
    sys.exit()

# ===============================================================================
# Train Loop
# ===============================================================================
if job_data['control_mode'] not in ['Torque', 'PID']:
    print('Unknown action space! Please check the parameter control_mode in the job script.')
    sys.exit()
if job_data['policy_output'] not in ['jp', 'djp']:
    print('Unknown policy output! Please check the parameter policy_output in the job script.')
    sys.exit()
if job_data['regularizer'] not in ['dapg', 'none']:
    print('Unknown regularizer type! Please check the parameter regularizer in the job script.')
    sys.exit()
if task_id == 'relocate':
    e = GymEnv(job_data['env'], job_data['control_mode'], job_data['object'])  # an unified env wrapper for all kind of envs
else:
    e = GymEnv(job_data['env'], job_data['control_mode'])  # an unified env wrapper for other envs
num_future_state = len(job_data['future_s'])
num_history_state = job_data['history_s'] # Note that if job_data['history_s'] sets to be a negative value, we do not add the history information
# Add both the histroy info (state + action) and the predicted states into the inputs 
# obs_dim:num_robot_s + num_object_s, act_dim: num_robot_s
Koopman_obser = DraftedObservable(num_robot_s, num_object_s)
if job_data['freeze_base']:  # only adapt the finger motions, or only include the rotation angles
    if job_data['include_Rots']:
        if task_id == 'door' or task_id == 'relocate':
            num_robot_s = 27 # include all rotation angles
        else:  # hammer task
            job_data['freeze_base'] = False  # there are only hand_Rxy for this task, so no need to adjust 'num_robot_s' -> use default num_robot_s = 26
    else:  
        # only adapt the finger motions
        # for the door opening task, we might add the hand_Rz (to make sure that the hand could generate enough torque to turn the handle)
        if task_id == 'door':
            # num_robot_s = 26  # include both hand_Tz and hand_Rz
            num_robot_s = 25  # only include hand_Rz
            # num_robot_s = 24  # only fingers
        else:
            num_robot_s = 24 # only fingers
if num_history_state >= 0: # add the history information into policy input
    if job_data['obj_dynamics']:
        observation_dim = (num_future_state + num_history_state + 1) * (num_robot_s + num_object_s) + (num_history_state + 1) * num_robot_s
    else:
        observation_dim = (num_future_state + num_history_state + 1) * num_robot_s + (num_history_state + 1) * num_robot_s
else:
    if job_data['obj_dynamics']:
        observation_dim = (num_future_state + 1) * (num_robot_s + num_object_s)
    else:
        observation_dim = (num_future_state + 1) * num_robot_s
print("EnvID: ", e.env_id)
print("observation dimension: ", observation_dim)
print("action dimension: ", num_robot_s)
print("horizon: ", task_horizon)
print("number of robot states:", num_robot_s)
print("number of object states:", num_object_s)
print("future states:",  job_data['future_s']) # goal states (at n future timesteps)
print("num of history states:",  job_data['history_s']) # goal states (at n history timesteps)
print("policy output:", job_data['policy_output']) # the type of policy output: jp->joint position; djp->delta joint position (residual policy)

policy = MLP(observation_dim = observation_dim, action_dim = num_robot_s, policy_output = job_data['policy_output'],
        hidden_sizes=job_data['policy_size'], seed=job_data['fixed_seed'], init_log_std = job_data['init_log_std'], min_log_std = job_data['min_log_std'], freeze_base=job_data['freeze_base'], include_Rots=job_data['include_Rots'])
baseline = MLPBaseline(inp_dim = observation_dim, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],hidden_sizes=job_data['vf_size'],
                       epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])  # the baseline model used
# demos = pickle.load(open(args.eval_data, 'rb'))
demos=pickle.load(open('/home/pshah479/Desktop/mjrl_repo/Relocation/Data/Relocate_task.pickle', 'rb'))
Eval_data = demo_playback(e, resnet_model, demos, len(demos), task_id)
coeffcients = dict()
coeffcients['task_ratio'] = job_data['task_ratio']
coeffcients['tracking_ratio'] = job_data['tracking_ratio']
coeffcients['hand_track'] = job_data['hand_track']
coeffcients['object_track'] = job_data['object_track']
coeffcients['ADD_BONUS_REWARDS'] = 1 # when evaluating the policy, it is always set to be enabled
coeffcients['ADD_BONUS_PENALTY'] = 1

# if job_data['eval_rollouts'] >= 1:  
#     # in this function, they also set the seed.
#     score = e.evaluate_policy(resnet_model, Eval_data, Simple_PID, coeffcients, Koopman_obser, KODex, task_horizon, job_data['future_s'], job_data['history_s'], policy, num_episodes=len(Eval_data), obj_dynamics = job_data['obj_dynamics'])  # noise-free actions
#     # score[0][0] -> mean rewards for rollout
#     print("On initial policy, (gamma = 1) mean task reward = %f, mean tracking reward = %f" % (score[0][0], score[0][4]))
pickle.dump(policy, open(JOB_DIR + '/Initial_policy_without_BC.pickle', 'wb'))
if args.pre_trained_path == 'none':  # training from scratch, instead of applying fine-tuning on new objects
    bc_agent = BC(JOB_DIR, Eval_data, policy=policy, refer_motion = KODex, Koopman_obser = Koopman_obser, task_id = task_id,
                object_dynamics = job_data['obj_dynamics'], robot_dim = num_robot_s, obj_dim = num_object_s,
                epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'], lr=job_data['bc_learn_rate'], loss_type='MSE')  # pre-training on the model policy
    # If we choose to normalize the model output or not. This would overwrite the given init_log_std
    if job_data['bc_normalize'] == 1:
        print("Normalize the NN input and output!")
        # compute the shift transformation using all reference motion
        demo_observation, demo_action, in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations(num_traj = len(Eval_data), future_s = job_data['future_s'], history_s = job_data['history_s'], task_horizon = task_horizon)
        # if the policy is learning in the joint space, its very important to have this transformations.
        # as seen in many locomotion papers, it is common and important to limit the predictions to be shifts.
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)  # this is equivalent to compute the default pose and a default scale
        if job_data['policy_output'] == 'djp':   # outputs are residual joint positions 'djp' -> so we always set it to be 'job_data['init_log_std']'
            out_scale[:] = np.exp(job_data['init_log_std'])  # out_scale -> action noise 
            # for the 6D poses of the hand, we might use default action noise, instead of out_scale.
        if job_data['freeze_base']:
            if job_data['include_Rots']: # if include all rotations, we might use smaller variance values
                if task_id == 'relocate' or task_id == 'door':
                    out_scale[:3] = np.exp(job_data['init_log_std'])  
                else:
                    out_scale[:2] = np.exp(job_data['init_log_std']) 
        else: # include all 6D poses of the hand:
            if task_id == 'door':
                out_scale[:4] = np.exp(job_data['init_log_std'])  
            elif task_id == 'hammer':
                out_scale[:2] = np.exp(job_data['init_log_std'])  
            elif task_id == 'relocate':
                out_scale[:6] = np.exp(job_data['init_log_std']) 
        bc_agent.set_variance_with_data(out_scale) # set self.log_std, which was initialized as job_data['init_log_std']
    ts = timer.time()
    num_traj = len(Eval_data)   
    if job_data['bc_traj'] != 0: # use specific number of trajs, instead of all demos
        num_traj = job_data['bc_traj']
    # Note BC only works when the control mode is set as 'PID'.
    if job_data['bc'] == 1 and job_data['control_mode'] == 'PID': # to warm_start the motion adapter via mimicking the reference motion (behavior guidance)
        print("========================================")
        print("Running BC for motion adapter with reference motion")
        print("========================================")
        data_augment_params = {'bc_augment': job_data['bc_augment'], 'bc_hand_noise': job_data['bc_hand_noise'], 'bc_obj_noise' : job_data['bc_obj_noise'], 'bc_aug_size': job_data['bc_aug_size']}
        bc_agent.train(save_policy=True, num_traj = num_traj, future_s = job_data['future_s'], history_s = job_data['history_s'], task_horizon = task_horizon, data_augment_params = data_augment_params, resnet_model=resnet_model)
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")
        # score = e.evaluate_policy(resnet_model, Eval_data, Simple_PID, coeffcients, Koopman_obser, KODex, task_horizon, job_data['future_s'], job_data['history_s'], policy, num_episodes=len(Eval_data), obj_dynamics = job_data['obj_dynamics'])  # noise-free actions
        # score[0][0] -> mean rewards for rollout
        # print("On BC policy, (gamma = 1) mean task reward = %f, mean tracking reward = %f" % (score[0][0], score[0][4])) 
else:  # loading the policies that are pre-trained on other objects and quick adapt on new objects
    print("load the policy from %s"%(args.pre_trained_path))
    with open(args.pre_trained_path, 'rb') as fp:
        policy = pickle.load(fp)
    baseline_path = (args.pre_trained_path[::-1].replace('policy'[::-1],'baseline'[::-1], 1))[::-1]
    pickle.dump(policy, open(JOB_DIR + '/Pre_trained_policy.pickle', 'wb'))
    with open(baseline_path, 'rb') as fp:
        baseline = pickle.load(fp)
    score = e.evaluate_policy(Eval_data, Simple_PID, coeffcients, Koopman_obser, KODex, task_horizon, job_data['future_s'], job_data['history_s'], policy, num_episodes=len(Eval_data), obj_dynamics = job_data['obj_dynamics'])  # noise-free actions
    # score[0][0] -> mean rewards for rollout
    print("On pre-trained policy, (gamma = 1) mean task reward = %f, mean tracking reward = %f" % (score[0][0], score[0][4]))

demo_path = None
if job_data['regularizer'] == 'dapg':  # add the necessary parameters to use this regularizer
    demo_path = dict()
    demo_path['observations'] = demo_observation
    demo_path['actions'] = demo_action
    demo_path['lam_0'] = job_data['lam_0']
    demo_path['lam_1'] = job_data['lam_1']
# ===============================================================================
# RL Loop
# ===============================================================================
if job_data['algorithm'] == 'NPG':
    rl_agent = NPG(e, policy, baseline,
                    normalized_step_size=job_data['rl_step_size'],
                    seed=job_data['RL_seed'], save_logs=True
                    )  
elif job_data['algorithm'] == 'PPO':
    rl_agent = PPO(e, policy, baseline, demo_path, 
                    clip_coef=job_data['PPO_clip'], epochs=job_data['PPO_epoch'],
                    mb_size=job_data['PPO_bs'],learn_rate=job_data['rl_step_size'],
                    seed=job_data['RL_seed'], save_logs=True, kl_target_diver=job_data['rl_desired_kl'], downscale=job_data['adv_scale']
                    )  # implementation-wise, PPO is simpler than others
print("========================================")
print("Starting reinforcement learning phase")
print("========================================")

ts = timer.time()
coeffcients['ADD_BONUS_REWARDS'] = job_data['ADD_BONUS_REWARDS'] # read it from the job file for training
coeffcients['ADD_BONUS_PENALTY'] = job_data['ADD_BONUS_PENALTY']

train_agent(job_name=JOB_DIR,
            agent=rl_agent,
            e=e,
            seed=job_data['fixed_seed'],  # fixed seed used in evaluation
            niter=job_data['rl_num_iter'], 
            gamma=job_data['rl_gamma'],  # GAE mode to compute the advantages
            gae_lambda=job_data['rl_gae'], # GAE mode to compute the advantages
            num_cpu=job_data['num_cpu'],
            sample_mode='trajectories',  # train_mode = trajectories
            num_traj=job_data['rl_num_traj'],
            save_freq=job_data['save_freq'],
            evaluation_rollouts=job_data['eval_rollouts'],
            task_horizon=task_horizon,
            future_state=job_data['future_s'],
            history_state=job_data['history_s'],
            Koopman_obser=Koopman_obser, 
            KODex=KODex,
            coeffcients=coeffcients,
            obj_dynamics=job_data['obj_dynamics'],
            control_mode=job_data['control_mode'],
            PD_controller=Simple_PID,
            resnet_model=resnet_model
            )
print("time taken = %f" % (timer.time()-ts))

print("Training finishes.")
