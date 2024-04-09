
import logging
logging.disable(logging.CRITICAL)

from tabulate import tabulate
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core_soil import sample_paths
import numpy as np
import pickle
import time as timer
import os
import copy

def train_agent(job_name, agent,
                e=None,
                seed = 0,
                niter = 101,
                gamma = 0.995,
                gae_lambda = None,
                num_cpu = 1,
                sample_mode = 'trajectories',
                num_traj = 50,
                save_freq = 10,
                evaluation_rollouts = None,
                plot_keys = ['stoc_pol_mean', 'eval_score', 'eval_success', 'success_rate'],
                ):
    # stoc_pol_mean -> not discounted mean task rewards
    np.random.seed(seed)
    if os.path.isdir(job_name) == False:
        os.mkdir(job_name)
    previous_dir = os.getcwd()
    os.chdir(job_name) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False and agent.save_logs == True: os.mkdir('logs')
    best_policy = copy.deepcopy(agent.policy)
    best_perf = -1e8
    train_curve = best_perf*np.ones(niter)
    mean_pol_perf = 0.0
    if e is None:
        e = GymEnv(agent.env.env_id, 'Torque')
    for i in range(niter):  # fresh start to 100 (100 iterations)
        print("......................................................................................")
        print("ITERATION : %i " % i)
        if train_curve[i-1] > best_perf:
            best_policy = copy.deepcopy(agent.policy)
            best_perf = train_curve[i-1]

        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths(num_traj=evaluation_rollouts, policy=agent.policy, num_cpu=num_cpu,
                                      env=e, eval_mode=True, base_seed=seed)
            mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])  # mean rewards
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_pol_perf)
                try:
                    eval_success = e.env.env.evaluate_success(eval_paths)  # success_percentage
                    agent.logger.log_kv('eval_success', eval_success)
                except:
                    pass

        N = num_traj
        args = dict(N=N, env=e, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu)

        # all the important training functions are included in batch_reinforce.py or dapg.py  (train_step function)
        stats = agent.train_step(**args)  # this function is in batch_reinforce.py, with a train_from_paths function that can be rewritten


        train_curve[i] = stats[0] # mean rewards obtained during training using the policy before update

        if i % save_freq == 0 and i > 0:
            if agent.save_logs:
                agent.logger.save_log('logs/')
                make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
            policy_file = 'policy_%i.pickle' % i
            baseline_file = 'baseline_%i.pickle' % i
            pickle.dump(agent.policy, open('iterations/' + policy_file, 'wb'))
            pickle.dump(agent.baseline, open('iterations/' + baseline_file, 'wb'))
            pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
            # pickle.dump(agent.global_status, open('iterations/global_status.pickle', 'wb'))

        # print results to console
        if i == 0:
            result_file = open('results.txt', 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
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
