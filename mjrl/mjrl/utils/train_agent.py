import logging
logging.disable(logging.CRITICAL)

from tabulate import tabulate
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core import sample_paths
import numpy as np
import pickle
import time as timer
import os
import copy
from tqdm import tqdm 

def _load_latest_policy_and_logs(agent, *, policy_dir, logs_dir):
    """Loads the latest policy.
    Returns the next step number to begin with.
    """
    assert os.path.isdir(policy_dir), str(policy_dir)
    assert os.path.isdir(logs_dir), str(logs_dir)

    log_csv_path = os.path.join(logs_dir, 'log.csv')
    if not os.path.exists(log_csv_path):
        return 0   # fresh start

    print("Reading: {}".format(log_csv_path))
    agent.logger.read_log(log_csv_path)
    last_step = agent.logger.max_len - 1
    if last_step <= 0:
        return 0   # fresh start


    # find latest policy/baseline
    i = last_step
    while i >= 0:
        policy_path = os.path.join(policy_dir, 'policy_{}.pickle'.format(i))
        baseline_path = os.path.join(policy_dir, 'baseline_{}.pickle'.format(i))

        if not os.path.isfile(policy_path):
            i = i -1
            continue
        else:
            print("Loaded last saved iteration: {}".format(i))

        with open(policy_path, 'rb') as fp:
            agent.policy = pickle.load(fp)
        with open(baseline_path, 'rb') as fp:
            agent.baseline = pickle.load(fp)

        # additional
        # global_status_path = os.path.join(policy_dir, 'global_status.pickle')
        # with open(global_status_path, 'rb') as fp:
        #     agent.load_global_status( pickle.load(fp) )

        agent.logger.shrink_to(i + 1)
        assert agent.logger.max_len == i + 1
        return agent.logger.max_len

    # cannot find any saved policy
    raise RuntimeError("Log file exists, but cannot find any saved policy.")

def train_agent(job_name, agent,
                e,
                seed = 0,
                niter = 101,
                gamma = 0.995,
                gae_lambda = None,
                num_cpu = 1,
                sample_mode = 'trajectories',
                num_traj = 50,
                num_samples = 50000, # has precedence, used with sample_mode = 'samples'
                save_freq = 10,
                evaluation_rollouts = None,
                plot_keys = ['time_sampling', 'stoc_pol_mean', 'rollout_mean_task_rewards', 'rollout_min_task_rewards', 'rollout_max_task_rewards', 'rollout_mean_tracking_rewards', 'rollout_min_tracking_rewards', 'rollout_max_tracking_rewards', 'discounted_traj_rewards', 'eval_score', 'eval_task_rewards', 'eval_tracking_rewards', 'mean_std', 'mean_log_std'],  # mean rewards during RL training
                task_horizon=1e6,
                future_state=(1,2,3),
                history_state=0,
                Koopman_obser=None, 
                KODex=None,
                coeffcients=None,
                obj_dynamics=True,
                control_mode='Torque',
                PD_controller=None,
                ):
    plot_keys.append('eval_tracking_hand_rewards')
    plot_keys.append('eval_tracking_object_rewards')
    plot_keys.append('rollout_mean_tracking_hand_rewards')
    plot_keys.append('rollout_min_tracking_hand_rewards')
    plot_keys.append('rollout_max_tracking_hand_rewards')
    plot_keys.append('rollout_mean_tracking_object_rewards')
    plot_keys.append('rollout_min_tracking_object_rewards')
    plot_keys.append('rollout_max_tracking_object_rewards')
    plot_keys.append('eval_success')
    plot_keys.append('rollout_success')
    coeffcients_eval = coeffcients.copy()
    coeffcients_eval['ADD_BONUS_REWARDS'] = 1 # when evaluating the policy, it is always set to be enabled
    coeffcients_eval['ADD_BONUS_PENALTY'] = 1
    if os.path.isdir(job_name) == False:
        os.mkdir(job_name)
    previous_dir = os.getcwd()
    os.chdir(job_name) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False and agent.save_logs == True: os.mkdir('logs')
    best_policy = copy.deepcopy(agent.policy)
    best_eval_policy_score = copy.deepcopy(agent.policy)
    best_eval_policy_sr = copy.deepcopy(agent.policy)
    best_perf = -1e8
    best_eval_perf_score = -1e8 
    best_eval_perf_sr = -1e8 
    train_curve = best_perf*np.ones(niter)
    eval_curve_score = best_eval_perf_score*np.ones(niter)
    eval_curve_sr = best_eval_perf_sr*np.ones(niter)
    mean_pol_perf = 0.0
    task = agent.env.env_id.split('-')[0]
    best_policy_index = 0
    best_eval_policy_index_score = 0
    best_eval_policy_index_sr = 0
    # Load from any existing checkpoint, policy, statistics, etc.
    # Why no checkpointing.. :(
    i_start = _load_latest_policy_and_logs(agent,
                                           policy_dir='iterations',
                                           logs_dir='logs')
    if i_start:
        print("Resuming from an existing job folder ...")

    for i in tqdm(range(i_start, niter)):  # fresh start to 100 (100 iterations)
        print("......................................................................................")
        print("ITERATION : %i " % i)
        if train_curve[i-1] > best_perf:
            best_policy_index = i-1
            best_policy = copy.deepcopy(agent.policy)
            best_perf = train_curve[i-1]

        if eval_curve_score[i-1] > best_eval_perf_score:
            best_eval_policy_index_score = i-1
            best_eval_policy_score = copy.deepcopy(agent.policy)
            best_eval_perf_score = eval_curve_score[i-1]
        if eval_curve_sr[i-1] > best_eval_perf_sr:
            best_eval_policy_index_sr = i-1
            best_eval_policy_sr = copy.deepcopy(agent.policy)
            best_eval_perf_sr = eval_curve_sr[i-1]

        N = num_traj if sample_mode == 'trajectories' else num_samples
        args = dict(N=N, sample_mode=sample_mode, horizon=task_horizon, future_state=future_state, history_state = history_state, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu, Koopman_obser=Koopman_obser,KODex=KODex,coeffcients=coeffcients,obj_dynamics=obj_dynamics,control_mode=control_mode,PD_controller=PD_controller)

        # Before RL training, we first run the evaluation on the warm-started policy
        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths(num_traj=evaluation_rollouts, policy=agent.policy, num_cpu=num_cpu,
                                        horizon=task_horizon, future_state=future_state, history_state=history_state,
                                      env=e, task_id = e.env_id.split('-')[0], eval_mode=True, base_seed=seed,
                                      Koopman_obser=Koopman_obser, KODex=KODex,coeffcients=coeffcients_eval,
                                      obj_dynamics=obj_dynamics,control_mode=control_mode,PD_controller=PD_controller)
            mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])  # (not discounted) mean combined rewards
            mean_task_rewards = np.mean([np.sum(path['task_rewards']) for path in eval_paths])  # (not discounted) mean task rewards
            mean_tracking_rewards = np.mean([np.sum(path['tracking_rewards']) for path in eval_paths])  # (not discounted) mean tracking rewards
            mean_tracking_hand_rewards = np.mean([np.sum(path['tracking_hand_rewards']) for path in eval_paths])  # (not discounted) mean tracking rewards
            mean_tracking_object_rewards = np.mean([np.sum(path['tracking_object_rewards']) for path in eval_paths])  # (not discounted) mean tracking rewards
            eval_curve_score[i] = mean_pol_perf
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_pol_perf)
                agent.logger.log_kv('eval_task_rewards', mean_task_rewards)
                agent.logger.log_kv('eval_tracking_rewards', mean_tracking_rewards)
                agent.logger.log_kv('eval_tracking_hand_rewards', mean_tracking_hand_rewards)
                agent.logger.log_kv('eval_tracking_object_rewards', mean_tracking_object_rewards)
                try:
                    eval_success = e.env.env.evaluate_success(eval_paths)  # success_percentage
                    agent.logger.log_kv('eval_success', eval_success)
                    eval_curve_sr[i] = eval_success  # if we prefer saving the eval policy based on the task success rate.
                except:
                    pass

        # all the important training functions are included in batch_reinforce.py or dapg.py  (train_step function)
        stats = agent.train_step(**args)  # this function is in batch_reinforce.py, with a train_from_paths function that can be rewritten
        # the stats are obtained before updating the policy (results on policy \pi_{t-1})

        train_curve[i] = stats[0] # mean sum rewards obtained during RL sampling using the policy before update
        
        if i % save_freq == 0 and i > 0:
            if agent.save_logs:
                agent.logger.save_log('logs/')
                make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
            policy_file = 'policy_%i.pickle' % i
            baseline_file = 'baseline_%i.pickle' % i
            pickle.dump(agent.policy, open('iterations/' + policy_file, 'wb'))
            pickle.dump(agent.baseline, open('iterations/' + baseline_file, 'wb'))
            pickle.dump(best_policy, open('iterations/best_policy_' + str(best_policy_index) + '.pickle', 'wb'))
            pickle.dump(best_eval_policy_score, open('iterations/best_eval_score_policy_' + str(best_eval_policy_index_score) + '.pickle', 'wb'))
            pickle.dump(best_eval_policy_sr, open('iterations/best_eval_sr_policy_' + str(best_eval_policy_index_sr) + '.pickle', 'wb'))
            # pickle.dump(agent.global_status, open('iterations/global_status.pickle', 'wb'))

        # print results to console
        if i == 0:
            result_file = open('results.txt', 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        # train_curve[i] -> mean summed rewards during RL (not discounted)
        # mean_pol_perf -> mean summed rewards during evaluation (not discounted)
        # best_perf -> best train_curve[i] (not discounted)
        print("[ %s ] %4i %5.2f %5.2f %5.2f " % (timer.asctime(timer.localtime(timer.time())),
                                                 i, train_curve[i], mean_pol_perf, best_perf))
        result_file = open('results.txt', 'a')
        result_file.write("%4i %5.2f %5.2f %5.2f \n" % (i, train_curve[i], mean_pol_perf, best_perf))
        result_file.close()
        if agent.save_logs:  # True
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                       agent.logger.get_current_log().items()))
            print(tabulate(print_data))

    # final save
    pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
    if agent.save_logs:
        agent.logger.save_log('logs/')
        make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
    os.chdir(previous_dir)
