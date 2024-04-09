"""
This is a job script for controller learning for KODex 1.0
"""

from re import A
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.algos.npg_cg import NPG
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
import matplotlib.pyplot as plt
import matplotlib as mpl

# using the recommendated fig params from https://github.com/jbmouret/matplotlib_for_papers#pylab-vs-matplotlib
fig_params = {
'axes.labelsize': 10,
'axes.titlesize':15,
'font.size': 10,
'legend.fontsize': 10,
'xtick.labelsize': 10,
'ytick.labelsize': 10,
'text.usetex': False,
'figure.figsize': [5, 4.5]
}
mpl.rcParams.update(fig_params)

def demo_playback(demo_paths, num_demo, task_id):
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
            observations = path['observations']  
            observations_visualize = path['observations_visualization']
            handVelocity = path['handVelocity'] 
            obs = observations[0] 
            obs_visual = observations_visualize[0]
            state_dict['init_states'] = path['init_state_dict']
            state_dict['handpos'] = obs[:30]
            state_dict['handvel'] = handVelocity[0][:30]
            objpos = obs[39:42] # in the world frame(on the table)
            state_dict['desired_pos'] = obs[45:48] 
            state_dict['objpos'] = objpos - obs[45:48] # converged object position
            state_dict['objorient'] = obs_visual[33:36]
            state_dict['objvel'] = handVelocity[0][30:]
        elif task_id == 'door':
            observations = path['observations']  
            observations_visualize = path['observations_visualization']
            obs = observations[0] 
            obs_visual = observations_visualize[0]
            state_dict['init_states'] = path['init_state_dict']
            state_dict['handpos'] = obs_visual[:28]
            state_dict['handvel'] = obs_visual[30:58]
            state_dict['objpos'] = obs[32:35]  # handle position
            state_dict['objvel'] = obs_visual[58:59]  # door hinge
            state_dict['handle_init'] = path['init_state_dict']['door_body_pos']
        elif task_id == 'hammer':
            observations = path['observations'] 
            handVelocity = path['handVelocity'] 
            obs = observations[0]
            allvel = handVelocity[0]
            state_dict['init_states'] = path['init_state_dict']
            state_dict['handpos'] = obs[:26]
            state_dict['handvel'] = allvel[:26]
            state_dict['objpos'] = obs[49:52] + obs[42:45] 
            state_dict['objorient'] = obs[39:42]
            state_dict['objvel'] = obs[27:33]
            state_dict['nail_goal'] = obs[46:49]
        Training_data.append(state_dict)
    print("Finish loading demo data!")
    return Training_data


# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data.')
parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
parser.add_argument('--policy', type=str, required=True, help='absolute path of the policy file')
parser.add_argument('--eval_data', type=str, required=True, help='absolute path to evaluation data')
parser.add_argument('--visualize', type=str, required=True, help='determine if visualizing the policy or not')
parser.add_argument('--save_fig', type=str, required=True, help='determine if saving all generated figures')
parser.add_argument('--only_record_video', type=str, required=False, default='False', help='determine if only recording the policy rollout')

args = parser.parse_args()
with open(args.config, 'r') as f:
    job_data = eval(f.read())
assert 'algorithm' in job_data.keys()
visualize = False
if args.visualize == "True":
    visualize = True
Save_fig = False
if args.save_fig == "True":
    Save_fig = True
Only_record_video = False
if args.only_record_video == "True":
    Only_record_video = True
assert any([job_data['algorithm'] == a for a in ['NPG', 'TRPO', 'PPO']])  # start from natural policy gradient for training
assert os.path.exists(os.getcwd() + job_data['matrix_file'] + job_data['env'].split('-')[0] + '/koopmanMatrix.npy')  # loading KODex reference dynamics
KODex = np.load(os.getcwd() + job_data['matrix_file'] + job_data['env'].split('-')[0] + '/koopmanMatrix.npy')

# ===============================================================================
# Set up the controller parameter
# ===============================================================================
# This set is used for control frequency: 500HZ (for all DoF)
PID_P = 10
PID_D = 0.005  
Simple_PID = PID(PID_P, 0.0, PID_D)
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
    num_object_s = 12
    task_horizon = 100
elif task_id == 'door':
    num_robot_s = 28
    num_object_s = 7
    task_horizon = 70
elif task_id == 'hammer':
    num_robot_s = 26
    num_object_s = 15
    task_horizon = 71  # stable motions after hit
else:
    print("Unkown task!")
    sys.exit()
# ===============================================================================
# Train Loop
# ===============================================================================
if job_data['control_mode'] not in ['Torque', 'PID']:
    print('Unknown action space! Please check the parameter control_mode in the job script.')
    sys.exit()
# Visualization
if task_id == 'relocate':
    e = GymEnv(job_data['env'], job_data['control_mode'], job_data['object'])  # an unified env wrapper for all kind of envs
else:
    e = GymEnv(job_data['env'], job_data['control_mode'])  # an unified env wrapper for other envs
policy = pickle.load(open(args.policy, 'rb'))  
Koopman_obser = DraftedObservable(num_robot_s, num_object_s)
demos = pickle.load(open(args.eval_data, 'rb'))
Eval_data = demo_playback(demos, len(demos), task_id)
coeffcients = dict()
coeffcients['task_ratio'] = job_data['task_ratio']
coeffcients['tracking_ratio'] = job_data['tracking_ratio']
coeffcients['hand_track'] = job_data['hand_track']
coeffcients['object_track'] = job_data['object_track']
coeffcients['ADD_BONUS_REWARDS'] = 1 # when evaluating the policy, it is always set to be enabled
coeffcients['ADD_BONUS_PENALTY'] = 1
gamma = 1.
print("gamma:", gamma)
try:
    policy.freeze_base
except:
    policy.freeze_base = False        
try:
    policy.include_Rots
except:
    policy.include_Rots = False   
print(policy.m)
# plug into a NN-based controller for Door task and test its performance, and also, tried to be compatible with the 24 DoFs training case.
# load a door opening controller, which is important to 
if not Only_record_video:
    if task_id == 'pen':
        score = e.evaluate_trained_policy(Eval_data, Simple_PID, coeffcients, Koopman_obser, KODex, task_horizon, job_data['future_s'], job_data['history_s'], policy, num_episodes=len(demos), gamma = gamma, obj_dynamics = job_data['obj_dynamics'], visual = visualize)  # noise-free actions
        # score[0][0] -> mean sum rewards for rollout (task-specific rewards)
        # because we are now in evaluation mode, the reward/score here is the sum of gamma_discounted rewards at t=0 (the sum rewards at other states at t=1,..,T are ignored) 
        # In other words, this can be seen as the reward in the domain of trajectory.
        print("(gamma = %f)\n KODex with PD: mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (gamma, score[0][0], score[0][4], score[0][8], score[0][9]))
        print("KODex with motion adapter (mean action): mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (score[1][0], score[1][4], score[1][8], score[1][9]))
        print("KODex with motion adapter (noisy action): mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (score[2][0], score[2][4], score[2][8], score[2][9]))
    elif task_id == 'door':
        score, R_z_motions = e.evaluate_trained_policy(Eval_data, Simple_PID, coeffcients, Koopman_obser, KODex, task_horizon, job_data['future_s'], job_data['history_s'], policy, num_episodes=len(demos), gamma = gamma, obj_dynamics = job_data['obj_dynamics'], visual = visualize)  # noise-free actions
        # score[0][0] -> mean sum rewards for rollout (task-specific rewards)
        # because we are now in evaluation mode, the reward/score here is the sum of gamma_discounted rewards at t=0 (the sum rewards at other states at t=1,..,T are ignored) 
        # In other words, this can be seen as the reward in the domain of trajectory.
        print("(gamma = %f)\n KODex with PD: mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (gamma, score[0][0], score[0][4], score[0][8], score[0][9]))
        print("KODex with motion adapter (mean action): mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (score[1][0], score[1][4], score[1][8], score[1][9]))
        print("KODex with motion adapter (noisy action): mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (score[2][0], score[2][4], score[2][8], score[2][9]))
        if Save_fig:  # study on how the motion get updated for the door task.
            root_dir = os.getcwd() + "/" + args.policy[:args.policy.find('.')]
            if not os.path.exists(root_dir):
                os.mkdir(root_dir)
            R_z_motion_MA, R_z_motion_PD = R_z_motions
            all_index = [i for i in range(len(Eval_data))]
            PD_motions = np.zeros([len(all_index), len(R_z_motion_PD[0])])
            MA_motions = np.zeros([len(all_index), len(R_z_motion_MA[0])])
            for i in range(len(all_index)):
                PD_motions[i, :] = np.array(R_z_motion_PD[all_index[i]])
                MA_motions[i, :] = np.array(R_z_motion_MA[all_index[i]])
            mean_ours, std_ours = np.mean(MA_motions, axis = 0), np.std(MA_motions, axis=0)
            mean_kodex_pd, std_kodex_pd = np.mean(PD_motions, axis = 0), np.std(PD_motions, axis=0)
            x_simu = np.arange(0, MA_motions.shape[1])
            fig, ax = plt.subplots()
            ax.plot(x_simu, mean_ours, linewidth=3, label = 'CIMER', color='purple')
            ax.fill_between(x_simu, mean_ours - std_ours, mean_ours + std_ours, alpha = 0.15, linewidth = 0, color='purple')
            ax.plot(x_simu, mean_kodex_pd, linewidth=3, label = 'Imitator + PD', color='b')
            ax.fill_between(x_simu, mean_kodex_pd - std_kodex_pd, mean_kodex_pd + std_kodex_pd, alpha = 0.15, linewidth = 0, color='b')
            ax.grid(True)
            plt.vlines(12, min(mean_ours), max(mean_ours), linewidth=3, linestyles='dotted', colors='k')  # around 12 iter, the hand begins to turn over the handle
            plt.vlines(30, min(mean_ours), max(mean_ours), linewidth=3, linestyles='dotted', colors='k')  # around 30 iter, the hand ends at turn over the handle
            # fig.suptitle("Changes of Rotation Angle", fontsize=22) #  for Handle Turning
            fig.supxlabel('Time Step', fontsize=20)
            fig.supylabel('Rotation Angle (rad)', fontsize=20)
            ax.tick_params(axis='both', labelsize=16)
            # legend = plt.legend(fontsize=10)
            # frame = legend.get_frame()
            # frame.set_facecolor('0.9')
            # frame.set_edgecolor('0.9')
            plt.tight_layout()
            plt.savefig(root_dir + '/rotation.png')
            plt.close()
    elif task_id == 'hammer':
        score, height_values = e.evaluate_trained_policy(Eval_data, Simple_PID, coeffcients, Koopman_obser, KODex, task_horizon, job_data['future_s'], job_data['history_s'], policy, num_episodes=len(demos), gamma = gamma, obj_dynamics = job_data['obj_dynamics'], visual = visualize)  # noise-free actions
        # score[0][0] -> mean sum rewards for rollout (task-specific rewards)
        # because we are now in evaluation mode, the reward/score here is the sum of gamma_discounted rewards at t=0 (the sum rewards at other states at t=1,..,T are ignored) 
        # In other words, this can be seen as the reward in the domain of trajectory.
        print("(gamma = %f)\n KODex with PD: mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (gamma, score[0][0], score[0][4], score[0][8], score[0][9]))
        print("KODex with motion adapter (mean action): mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (score[1][0], score[1][4], score[1][8], score[1][9]))
        print("KODex with motion adapter (noisy action): mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (score[2][0], score[2][4], score[2][8], score[2][9]))
        if Save_fig:
            root_dir = os.getcwd() + "/" + args.policy[:args.policy.find('.')]
            if not os.path.exists(root_dir):
                os.mkdir(root_dir)
            hammer_compare_index, MA_PD_plam_pos, PD_plam_pos, Joint_adaptations, PD_hit_pos, MA_hit_pos, PD_hit_force, MA_hit_force = height_values
            # post-process the data
            # plot a hist gram
            print(len(PD_hit_pos))
            print(len(MA_hit_pos))
            for z in range(2):
                if z == 0:
                    scatter_PD = np.zeros([len(PD_hit_pos)])
                    scatter_MA = np.zeros([len(MA_hit_pos)])
                    for i in range(len(PD_hit_pos)):
                        x = PD_hit_pos[i][0]
                        y = PD_hit_pos[i][1]
                        zz = PD_hit_pos[i][2]
                        scatter_PD[i] = np.sqrt(x ** 2 + y ** 2 + zz ** 2)
                    for i in range(len(MA_hit_pos)):
                        x = MA_hit_pos[i][0]
                        y = MA_hit_pos[i][1]
                        zz = MA_hit_pos[i][2]
                        scatter_MA[i] = np.sqrt(x ** 2 + y ** 2 + zz ** 2)
                    scale = 1000
                elif z == 1:
                    scatter_PD = np.zeros([len(PD_hit_force)])
                    scatter_MA = np.zeros([len(MA_hit_force)])
                    for i in range(len(PD_hit_force)):
                        scatter_PD[i] = PD_hit_force[i]
                    for i in range(len(MA_hit_force)):
                        scatter_MA[i] = MA_hit_force[i]
                    scale = 1
                plt.figure(z)
                fig, ax = plt.subplots()
                num_bins = 40
                n, bins, patches = ax.hist(scatter_MA * scale, num_bins, color='purple', alpha=0.7, label="CIMER")
                mu = np.mean(scatter_MA * scale)
                sigma = np.std(scatter_MA * scale)
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                    np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) 
                aaa = np.histogram(scatter_MA, num_bins)
                y = y / max(y) * max(aaa[0])
                # plt.plot(bins, y, '--', color ='purple')
                n, bins, patches = ax.hist(scatter_PD * scale, num_bins, color='b', alpha=0.7, label="Motion Gen. + PD")
                mu = np.mean(scatter_PD * scale)
                sigma = np.std(scatter_PD * scale)
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                    np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
                a = np.histogram(scatter_PD, num_bins)
                y = y / max(y) * max(a[0])
                # plt.plot(bins, y, '--', color ='b')
                # ax.legend(loc="upper right")
                if z == 0:
                    ax.set_ylim([0, 1.2 * max(max(a[0]), max(aaa[0]))])
                    ax.set_xlim([0.8 * min(min(scatter_MA * scale), min(scatter_PD * scale)), 1.1 * max(max(scatter_MA * scale), max(scatter_PD * scale))])
                elif z == 1:
                    ax.set_ylim([0, 1.2 * max(max(a[0]), max(aaa[0]))])
                    ax.set_xlim([0.8 * min(min(scatter_PD * scale), min(scatter_MA * scale)), 1.1 * max(max(scatter_MA * scale), max(scatter_PD * scale))])
                ax.grid(True)
                if z == 0:
                    # fig.suptitle("Changes of Contact Point", fontsize=22)
                    fig.supxlabel('Distance from contact point to center', fontsize=18)
                    fig.supylabel('Occurrence', fontsize=20)
                    # legend = plt.legend(fontsize=18)
                    ax.tick_params(axis='both', labelsize=16)
                    # frame = legend.get_frame()
                    # frame.set_facecolor('0.9')
                    # frame.set_edgecolor('0.9')
                    plt.tight_layout()
                    plt.savefig(root_dir + '/%s.png'%("hit_location"))
                    plt.close()
                elif z == 1:
                    # fig.suptitle("Changes of Contact Force", fontsize=22)
                    fig.supxlabel('Contact Force', fontsize=18)
                    fig.supylabel('Occurrence', fontsize=20)
                    # legend = plt.legend(fontsize=18)
                    ax.tick_params(axis='both', labelsize=16)
                    # frame = legend.get_frame()
                    # frame.set_facecolor('0.9')
                    # frame.set_edgecolor('0.9')
                    plt.tight_layout()
                    plt.savefig(root_dir + '/%s.png'%("contact_force"))
                    plt.close()
            ''' old visulization codes
            all_index = [i for i in range(len(Eval_data))]
            hammer_compare_index = all_index # compare the performance across all samples
            Joint_changes = np.zeros([num_robot_s, task_horizon - 1])
            # Joints_name = {"Base":["A_ARRx", "A_ARRy"], "Wrist":["A_WRJ1", "A_WRJ0"], "Forefinger":["A_FFJ3", "A_FFJ2", "A_FFJ1", "A_FFJ0"], "MiddleFinger":["A_MFJ3", "A_MFJ2", "A_MFJ1", "A_MFJ0"], "RingFinger":["A_RFJ3", "A_RFJ2", "A_RFJ1", "A_RFJ0"], "LittleFinger":["A_LFJ4", "A_LFJ3", "A_LFJ2", "A_LFJ1", "A_LFJ0"], "Thumb":["A_THJ4", "A_THJ3", "A_THJ2", "A_THJ1", "A_THJ0"]}
            Joints_name = {"Base":["Rx", "Ry"], "Wrist":["WRJ1", "WRJ0"], "Forefinger":["FFJ3", "FFJ2", "FFJ1", "FFJ0"], "MiddleFinger":["MFJ3", "MFJ2", "MFJ1", "MFJ0"], "RingFinger":["RFJ3", "RFJ2", "RFJ1", "RFJ0"], "LittleFinger":["LFJ4", "LFJ3", "LFJ2", "LFJ1", "LFJ0"], "Thumb":["THJ4", "THJ3", "THJ2", "THJ1", "THJ0"]}
            for i in range(len(hammer_compare_index)):
                for j in range(len(Joint_adaptations[hammer_compare_index[i]])):
                    Joint_changes[:, j] += Joint_adaptations[hammer_compare_index[i]][j] / len(hammer_compare_index)
            finger_index = 0
            for i  in range(len(Joints_name.keys())):
                body_type = list(Joints_name.keys())[i]
                if body_type not in {"Wrist", "Base"}:  # Wrist only has two parts, so one row is enough
                    plt.figure(i)
                    fig, ax = plt.subplots(2, 3)
                    for j in range(len(Joints_name[body_type])):
                        ax[j // 3 , j % 3].plot(Joint_changes[finger_index], linewidth=1, color='#B22400')
                        # ax[j // 3 , j % 3].vlines(11, 1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index]), linestyles='dotted', colors='k')
                        ax[j // 3 , j % 3].set_ylim([1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index])])
                        ax[j // 3 , j % 3].set(title=Joints_name[body_type][j])
                        finger_index += 1
                        # fig.legend(['Differences of joint targets (%s) made by MA'%(body_type)], loc='lower left')
                        # fig.legend(['Joint '], loc='lower left')
                        fig.supxlabel('Time step')
                        fig.supylabel('Differences of joint targets')
                        ax[j // 3 , j % 3].grid()
                else:
                    plt.figure(i)
                    fig, ax = plt.subplots(1, 2)
                    for j in range(2):
                        ax[j % 2].plot(Joint_changes[finger_index], linewidth=1, color='#B22400')
                        # ax[j % 2].vlines(11, 1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index]), linestyles='dotted', colors='k')
                        ax[j % 2].set_ylim([1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index])])
                        ax[j % 2].set(title=Joints_name[body_type][j])
                        finger_index += 1
                        # fig.legend(['Differences of joint targets (%s) made by MA'%(body_type)], loc='lower left')
                        fig.supxlabel('Time step')
                        fig.supylabel('Differences of joint targets')
                        ax[j % 2].grid()
                plt.tight_layout()
                plt.savefig(root_dir + '/%s.png'%(body_type))
                plt.close()
            # all_index = [i for i in range(len(Eval_data))]
            # hammer_compare_index = all_index # compare the performance across all samples
            MA_PD_all_height = np.zeros(task_horizon - 1, )
            PD_all_height = np.zeros(task_horizon - 1)
            ours_all_height = np.zeros([task_horizon - 1, len(hammer_compare_index)])
            KODex_PD_all_height = np.zeros([task_horizon - 1, len(hammer_compare_index)])
            for i in range(len(hammer_compare_index)):
                plt.figure(i)
                fig, ax = plt.subplots()
                index_ = 0
                for item1, item2 in zip(MA_PD_plam_pos[hammer_compare_index[i]], PD_plam_pos[hammer_compare_index[i]]):
                    MA_PD_all_height[index_] += item1 / len(hammer_compare_index)
                    PD_all_height[index_] += item2 / len(hammer_compare_index)
                    ours_all_height[index_, i] = item1
                    KODex_PD_all_height[index_, i] = item2
                    index_ += 1
                ax.plot(MA_PD_plam_pos[hammer_compare_index[i]], linewidth=1, color='#B22400')
                ax.plot(PD_plam_pos[hammer_compare_index[i]], linewidth=1, color='#F22BB2')
                vis_min = min(min(MA_PD_plam_pos[hammer_compare_index[i]]), min(PD_plam_pos[hammer_compare_index[i]]))
                vis_max = max(max(MA_PD_plam_pos[hammer_compare_index[i]]), max(PD_plam_pos[hammer_compare_index[i]]))
                ax.vlines(11, 1.1 * vis_min, 1.1 * vis_max, linestyles='dotted', colors='k')  # around 11 iter, the hand grasps the hammer
                ax.set_ylim([1.1 * vis_min, 1.1 * vis_max])
                ax.set(title="plam_height - nail_height(index:" + str(hammer_compare_index[i]) + ")")
                fig.legend(['MA_PD', 'PD'], loc='lower left')
                fig.supxlabel('Time step')
                fig.supylabel('Height Difference')
                plt.tight_layout()
                plt.savefig(root_dir + '/height_diff_' + str(hammer_compare_index[i]) + '.png')
                plt.close()
            x_simu = np.arange(0, ours_all_height.shape[0])
            plt.figure(200)
            plt.axes(frameon=0)
            plt.grid()
            mean_ours, std_ours = np.mean(ours_all_height, axis = 1), np.std(ours_all_height, axis=1)
            mean_kodex_pd, std_kodex_pd = np.mean(KODex_PD_all_height, axis = 1), np.std(KODex_PD_all_height, axis=1)
            plt.plot(x_simu, mean_ours, linewidth=2, label = 'KODex + Motion Adapter + PD', color='#B22400')
            plt.fill_between(x_simu, mean_ours - std_ours, mean_ours + std_ours, alpha = 0.15, linewidth = 0, color='#B22400')
            plt.plot(x_simu, mean_kodex_pd, linewidth=2, label = 'KODex + PD', color='#006BB2')
            plt.fill_between(x_simu, mean_kodex_pd - std_kodex_pd, mean_kodex_pd + std_kodex_pd, alpha = 0.15, linewidth = 0, color='#006BB2')
            plt.vlines(23, -0.15, 0.05, linestyles='dotted', colors='k')  # around 11 iter, the hand grasps the hammer
            plt.xlabel('Time step', fontsize=12)
            plt.ylabel('Palm height - Nail height', fontsize=12)
            legend = plt.legend(fontsize=12)
            frame = legend.get_frame()
            frame.set_facecolor('0.9')
            frame.set_edgecolor('0.9')
            plt.tight_layout()
            plt.savefig(root_dir + '/all_height.png')
            '''
    elif task_id == 'relocate':
        score, force_values = e.evaluate_trained_policy(Eval_data, Simple_PID, coeffcients, Koopman_obser, KODex, task_horizon, job_data['future_s'], job_data['history_s'], policy, num_episodes=len(demos), gamma = gamma, obj_dynamics = job_data['obj_dynamics'], visual = visualize)  # noise-free actions
        # score[0][0] -> mean sum rewards for rollout (task-specific rewards)
        # because we are now in evaluation mode, the reward/score here is the sum of gamma_discounted rewards at t=0 (the sum rewards at other states at t=1,..,T are ignored) 
        # In other words, this can be seen as the reward in the domain of trajectory.
        print("(gamma = %f)\n KODex with PD: mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (gamma, score[0][0], score[0][4], score[0][8], score[0][9]))
        print("KODex with motion adapter (mean action): mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (score[1][0], score[1][4], score[1][8], score[1][9]))
        print("KODex with motion adapter (noisy action): mean task reward = %f, mean total tracking reward = %f, mean hand tracking reward = %f, mean object tracking reward = %f" % (score[2][0], score[2][4], score[2][8], score[2][9]))
        if Save_fig:
            root_dir = os.getcwd() + "/" + args.policy[:args.policy.find('.')]
            if not os.path.exists(root_dir):
                os.mkdir(root_dir)
            force_compare_index, MA_PD_tips_force, PD_tips_force, Joint_adaptations = force_values
            all_index = [i for i in range(len(Eval_data))]
            force_compare_index = all_index # compare the performance across all samples
            Joint_changes = np.zeros([num_robot_s, task_horizon - 1])
            Joints_name = {"Base":["A_ARTx", "A_ARTy", "A_ARTz", "A_ARRx", "A_ARRy", "A_ARRz"], "Wrist":["A_WRJ1", "A_WRJ0"], "Forefinger":["A_FFJ3", "A_FFJ2", "A_FFJ1", "A_FFJ0"], "MiddleFinger":["A_MFJ3", "A_MFJ2", "A_MFJ1", "A_MFJ0"], "RingFinger":["A_RFJ3", "A_RFJ2", "A_RFJ1", "A_RFJ0"], "LittleFinger":["A_LFJ4", "A_LFJ3", "A_LFJ2", "A_LFJ1", "A_LFJ0"], "Thumb":["A_THJ4", "A_THJ3", "A_THJ2", "A_THJ1", "A_THJ0"]}
            for i in range(len(force_compare_index)):
                for j in range(len(Joint_adaptations[force_compare_index[i]])):
                    Joint_changes[:, j] += Joint_adaptations[force_compare_index[i]][j] / len(force_compare_index)
            finger_index = 0
            for i in range(len(Joints_name.keys())):
                body_type = list(Joints_name.keys())[i]
                if body_type != "Wrist":  # Wrist only has two parts, so one row is enough
                    plt.figure(i)
                    fig, ax = plt.subplots(2, 3)
                    for j in range(len(Joints_name[body_type])):
                        ax[j // 3 , j % 3].plot(Joint_changes[finger_index], linewidth=1, color='#B22400')
                        ax[j // 3 , j % 3].vlines(22, 1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index]), linestyles='dotted', colors='k')
                        ax[j // 3 , j % 3].set_ylim([1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index])])
                        ax[j // 3 , j % 3].set(title=Joints_name[body_type][j])
                        finger_index += 1
                        fig.legend(['Differences of joint targets (%s) made by MA'%(body_type)], loc='lower left')
                        fig.supxlabel('Time step')
                        fig.supylabel('Differences of joint targets')
                else:
                    plt.figure(i)
                    fig, ax = plt.subplots(1, 2)
                    for j in range(2):
                        ax[j % 2].plot(Joint_changes[finger_index], linewidth=1, color='#B22400')
                        ax[j % 2].vlines(22, 1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index]), linestyles='dotted', colors='k')
                        ax[j % 2].set_ylim([1.1 * min(Joint_changes[finger_index]), 1.1 * max(Joint_changes[finger_index])])
                        ax[j % 2].set(title=Joints_name[body_type][j])
                        finger_index += 1
                        fig.legend(['Differences of joint targets (%s) made by MA'%(body_type)], loc='lower left')
                        fig.supxlabel('Time step')
                        fig.supylabel('Differences of joint targets')
                plt.tight_layout()
                plt.savefig(root_dir + '/%s.png'%(body_type))
                plt.close()
            # all_index = [i for i in range(len(Eval_data))]
            # force_compare_index = all_index # compare the performance across all samples
            finger_index = ['ff', 'mf', 'rf', 'lf', 'th', 'sum']
            finger_index_vis = ['Forefinger', 'Middlefinger', 'Ringfinger', 'Littlefinger', 'Thumb', 'Sum']
            MA_PD_total_force = {'ff': np.zeros(task_horizon - 1), 'mf': np.zeros(task_horizon - 1), 'rf': np.zeros(task_horizon - 1), 'lf': np.zeros(task_horizon - 1), 'th': np.zeros(task_horizon - 1), 'sum': np.zeros(task_horizon - 1)}
            PD_total_force = {'ff': np.zeros(task_horizon - 1), 'mf': np.zeros(task_horizon - 1), 'rf': np.zeros(task_horizon - 1), 'lf': np.zeros(task_horizon - 1), 'th': np.zeros(task_horizon - 1), 'sum': np.zeros(task_horizon - 1)}
            MA_PD_PD_ratio = {'ff': np.zeros(task_horizon - 1), 'mf': np.zeros(task_horizon - 1), 'rf': np.zeros(task_horizon - 1), 'lf': np.zeros(task_horizon - 1), 'th': np.zeros(task_horizon - 1), 'sum': np.zeros(task_horizon - 1)}
            # additional visualization codes
            ours_all_force = {'ff': np.zeros([task_horizon - 1, len(force_compare_index)]), 'mf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'rf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'lf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'th': np.zeros([task_horizon - 1, len(force_compare_index)]), 'sum': np.zeros([task_horizon - 1, len(force_compare_index)])}
            KODex_PD_all_force = {'ff': np.zeros([task_horizon - 1, len(force_compare_index)]), 'mf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'rf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'lf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'th': np.zeros([task_horizon - 1, len(force_compare_index)]), 'sum': np.zeros([task_horizon - 1, len(force_compare_index)])}
            ours_KODex_PD_all_force = {'ff': np.zeros([task_horizon - 1, len(force_compare_index)]), 'mf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'rf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'lf': np.zeros([task_horizon - 1, len(force_compare_index)]), 'th': np.zeros([task_horizon - 1, len(force_compare_index)]), 'sum': np.zeros([task_horizon - 1, len(force_compare_index)])}
            # additional visualization codes
            for i in range(len(force_compare_index)):
                plt.figure(i)
                fig, ax = plt.subplots(2, 3)
                for row in range(2):
                    for col in range(3):
                        tmp_length = len(MA_PD_tips_force[finger_index[3*row+col]][force_compare_index[i]])  # -> a list through time horizon
                        MA_PD_minus_PD = list()
                        index_ = 0
                        for item1, item2 in zip(MA_PD_tips_force[finger_index[3*row+col]][force_compare_index[i]], PD_tips_force[finger_index[3*row+col]][force_compare_index[i]][:tmp_length]):
                            MA_PD_total_force[finger_index[3*row+col]][index_] += item1 / len(force_compare_index)
                            PD_total_force[finger_index[3*row+col]][index_] += item2 / len(force_compare_index)
                            # additional visualization codes
                            ours_all_force[finger_index[3*row+col]][index_, i] += item1 
                            KODex_PD_all_force[finger_index[3*row+col]][index_, i] += item2 
                            if item1 / (item2 + 1e-8) > 5:
                                ours_KODex_PD_all_force[finger_index[3*row+col]][index_, i] = 5  
                            else:
                                ours_KODex_PD_all_force[finger_index[3*row+col]][index_, i] = item1 / (item2 + 1e-8)
                            # additional visualization codes
                            MA_PD_minus_PD.append(item1 - item2)
                            index_ += 1
                        ax[row, col].plot(MA_PD_minus_PD, linewidth=1, color='#B22400')
                        # ax[row, col].plot(PD_tips_force[finger_index[3*row+col]][force_compare_index[i]], linewidth=1, color='#F22BB2')
                        ax[row, col].vlines(22, 1.1 * min(MA_PD_minus_PD), 1.1 * max(MA_PD_minus_PD), linestyles='dotted', colors='k')
                        ax[row, col].set_ylim([1.1 * min(MA_PD_minus_PD), 1.1 * max(MA_PD_minus_PD)])
                        ax[row, col].set(title=finger_index[3*row+col])
                        # ax[row, col].legend()
                fig.legend(['MA_PD - PD (larger the value, fingertips are closer to the object)'], loc='lower left')
                fig.supxlabel('Time step')
                fig.supylabel('Difference of touch sensor feedback')
                plt.tight_layout()
                plt.savefig(root_dir + '/touch_sensor_' + str(force_compare_index[i]) + '.png')
                plt.close()

            for finger_ in finger_index:
                for i in range(task_horizon - 1):
                    if MA_PD_total_force[finger_][i] / (PD_total_force[finger_][i] + 1e-8) > 5:
                        MA_PD_PD_ratio[finger_][i] = 5
                    else:
                        MA_PD_PD_ratio[finger_][i] = MA_PD_total_force[finger_][i] / (PD_total_force[finger_][i] + 1e-8)
                    
            # plt.figure(1)
            # fig, ax = plt.subplots(2, 3)
            # for row in range(2):
            #     for col in range(3):
            #         ax[row, col].plot(MA_PD_PD_ratio[finger_index[3*row+col]], linewidth=1, color='#B22400')
            #         ax[row, col].set(title=finger_index_vis[3*row+col])
            #         ax[row, col].vlines(22, 0, 1.1 * max(MA_PD_PD_ratio[finger_index[3*row+col]]), linestyles='dotted', colors='k')
            # fig.legend(['Ratio of MA_PD over PD (max: 5)'], loc='lower left')
            # plt.tight_layout()
            # plt.savefig(root_dir + '/ratio.png')

            # additional visualization codes
            plt.figure(2) # plot values 
            fig, ax = plt.subplots(2, 3)
            for row in range(2):
                for col in range(3):
                    x_simu = np.arange(0, ours_all_force[finger_index[3*row+col]].shape[0])
                    low_ours, mid_ours, high_ours = np.percentile(ours_all_force[finger_index[3*row+col]], [25, 50, 75], axis=1)
                    low_kodex_pd, mid_kodex_pd, high_kodex_pd = np.percentile(KODex_PD_all_force[finger_index[3*row+col]], [25, 50, 75], axis=1)
                    mean_ours, std_ours = np.mean(ours_all_force[finger_index[3*row+col]], axis = 1), np.std(ours_all_force[finger_index[3*row+col]], axis=1)
                    mean_kodex_pd, std_kodex_pd = np.mean(KODex_PD_all_force[finger_index[3*row+col]], axis = 1), np.std(KODex_PD_all_force[finger_index[3*row+col]], axis=1)
                    ax[row, col].plot(x_simu, mean_ours, linewidth = 2, label = 'CIMER', color='purple')
                    ax[row, col].fill_between(x_simu, mean_ours - std_ours, mean_ours + std_ours, alpha = 0.15, linewidth = 0, color='purple')
                    ax[row, col].plot(x_simu, mean_kodex_pd, linewidth = 2, label = 'Imitator + PD', color='b')
                    ax[row, col].fill_between(x_simu, mean_kodex_pd - std_kodex_pd, mean_kodex_pd + std_kodex_pd, alpha = 0.15, linewidth = 0, color='b')
                    ax[row, col].set(title=finger_index_vis[3*row+col])
                    ax[row, col].grid()
                    ax[row, col].tick_params(axis='both', labelsize=16)
            # fig.suptitle("Changes of Grasping Force", fontsize=22)
            fig.supxlabel('Time Step', fontsize=20)
            fig.supylabel('Grasping Force', fontsize=20)
            # legend = fig.legend(fontsize=10, loc='lower left')
            # legend = ax[row, col].legend()
                    # ax[row, col].vlines(22, 0, 1.1 * max(MA_PD_PD_ratio[finger_index[3*row+col]]), linestyles='dotted', colors='k')
            # fig.legend(['Ratio of MA_PD over PD (max: 5)'], loc='lower left')
            plt.tight_layout()
            plt.savefig(root_dir + '/values.png')

            # plt.figure(3)  # plot ratios
            # fig, ax = plt.subplots(2, 3)
            # for row in range(2):
            #     for col in range(3):
            #         x_simu = np.arange(0, ours_KODex_PD_all_force[finger_index[3*row+col]].shape[0])
            #         mean_ratio, std_ratio = np.mean(ours_KODex_PD_all_force[finger_index[3*row+col]], axis = 1), np.std(ours_KODex_PD_all_force[finger_index[3*row+col]], axis=1)
            #         ax[row, col].plot(x_simu, mean_ratio, linewidth = 2, label = 'Ratio', color='#B22400')
            #         ax[row, col].fill_between(x_simu, mean_ratio - std_ratio, mean_ratio + std_ratio, alpha = 0.15, linewidth = 0, color='#B22400')
            #         ax[row, col].set(title=finger_index_vis[3*row+col])
            #         ax[row, col].grid()
            #         # legend = ax[row, col].legend()
            #         # ax[row, col].vlines(22, 0, 1.1 * max(MA_PD_PD_ratio[finger_index[3*row+col]]), linestyles='dotted', colors='k')
            # # fig.legend(['Ratio of MA_PD over PD (max: 5)'], loc='lower left')
            # plt.tight_layout()
            # plt.savefig(root_dir + '/ratio_new.png')

            # for finger_ in finger_index: # plot individual values 
            #     x_simu = np.arange(0, ours_all_force[finger_].shape[0])
            #     plt.figure(10)
            #     plt.axes(frameon=0)
            #     mean_ours, std_ours = np.mean(ours_all_force[finger_], axis = 1), np.std(ours_all_force[finger_], axis=1)
            #     mean_kodex_pd, std_kodex_pd = np.mean(KODex_PD_all_force[finger_], axis = 1), np.std(KODex_PD_all_force[finger_], axis=1)
            #     plt.plot(x_simu, mean_ours, linewidth=2, label = 'KODex + Motion Adapter + PD', color='#B22400')
            #     plt.fill_between(x_simu, mean_ours - std_ours, mean_ours + std_ours, alpha = 0.15, linewidth = 0, color='#B22400')
            #     plt.plot(x_simu, mean_kodex_pd, linewidth=2, label = 'KODex + PD', color='#006BB2')
            #     plt.fill_between(x_simu, mean_kodex_pd - std_kodex_pd, mean_kodex_pd + std_kodex_pd, alpha = 0.15, linewidth = 0, color='#006BB2')
            #     vis_min = min(min(mean_ours - std_ours), min(mean_kodex_pd - std_kodex_pd))
            #     vis_max = max(max(mean_ours + std_ours), max(mean_kodex_pd + std_kodex_pd))
            #     plt.vlines(25, vis_min, vis_max, linestyles='dotted', colors='k')  # denote the grasping process
            #     plt.vlines(33, vis_min, vis_max, linestyles='dotted', colors='k')  
            #     plt.grid()
            #     plt.xlabel('Time step', fontsize=12)
            #     plt.ylabel('Touch Sensor Feedback', fontsize=12)
            #     # legend = plt.legend(fontsize=12)
            #     # frame = legend.get_frame()
            #     # frame.set_facecolor('0.9')
            #     # frame.set_edgecolor('0.9')
            #     plt.tight_layout()
            #     plt.savefig(root_dir + '/value_%s.png'%(finger_))
            #     plt.close()

            # for finger_ in finger_index: # plot individual ratios 
            #     x_simu = np.arange(0, ours_KODex_PD_all_force[finger_].shape[0])
            #     plt.figure(11)
            #     plt.axes(frameon=0)
            #     mean_ours, std_ours = np.mean(ours_KODex_PD_all_force[finger_], axis = 1), np.std(ours_KODex_PD_all_force[finger_], axis=1)
            #     plt.plot(x_simu, mean_ours, linewidth=2, label = 'Ratio', color='#B22400')
            #     plt.fill_between(x_simu, mean_ours - std_ours, mean_ours + std_ours, alpha = 0.15, linewidth = 0, color='#B22400')
            #     vis_min = min(mean_ours - std_ours)
            #     vis_max = max(mean_ours + std_ours)
            #     plt.vlines(25, vis_min, vis_max, linestyles='dotted', colors='k')  # denote the grasping process
            #     plt.vlines(33, vis_min, vis_max, linestyles='dotted', colors='k')  
            #     plt.grid()
            #     plt.xlabel('Time step', fontsize=12)
            #     plt.ylabel('Touch Feedback Ratio', fontsize=12)
            #     legend = plt.legend(fontsize=12)
            #     # frame = legend.get_frame()
            #     # frame.set_facecolor('0.9')
            #     # frame.set_edgecolor('0.9')
            #     plt.tight_layout()
            #     plt.savefig(root_dir + '/ratio_new_%s.png'%(finger_))
            #     plt.close()
else:  # Only_record_video
    if task_id == 'relocate':
        e.Visualze_CIMER_policy(Eval_data, Simple_PID, coeffcients, Koopman_obser, KODex, task_horizon, job_data['future_s'], job_data['history_s'], policy, num_episodes=len(demos), gamma = gamma, obj_dynamics = job_data['obj_dynamics'], visual = visualize, object_name = job_data['object'])  # noise-free actions
    else:
        e.Visualze_CIMER_policy(Eval_data, Simple_PID, coeffcients, Koopman_obser, KODex, task_horizon, job_data['future_s'], job_data['history_s'], policy, num_episodes=len(demos), gamma = gamma, obj_dynamics = job_data['obj_dynamics'], visual = visualize)  # noise-free actions        