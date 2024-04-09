"""
Visualize the recorded errors for comparing the performance of using different number of observables
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import re
import numpy as np

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n+1)

def plot_curves(path_to_file):
    if not os.path.exists(path_to_file):
        print("File does not exist!")
        sys.exit()
    else:
        fig_params = {
        'axes.labelsize': 10,
        'font.size': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [5, 4.5]
        }
        mpl.rcParams.update(fig_params)
        fig_path_tmp = [ele + '/' for ele in path_to_file.split('/')]
        fig_path_simu_goal_demo = ''.join(fig_path_tmp[:-1]) + 'RolloutError_simu_demo(GoalError).png'
        f = open(path_to_file, 'r')
        error_files = f.readlines()
        koopman_erros = []
        labels = ["348", "72", "384", "936", "O5", "O6"] # compare the performance when choosing various number of observables
        for i in range(len(error_files)):
            print("This is %d test out of %d."%(i+1, len(error_files)))
            error = np.load(error_files[i][:-1])
            koopman_erros.append(error)
        x_simu = np.arange(0, koopman_erros[0].shape[0])
        plt.figure(2)
        plt.axes(frameon=0)
        plt.grid()
        cmap = get_cmap(len(koopman_erros))
        for i in range(len(koopman_erros)):
            plt.plot(x_simu, np.median(koopman_erros[i], axis = 1), linewidth=2, label = labels[i], color=cmap(i))
            plt.fill_between(x_simu, np.percentile(koopman_erros[i], 25, axis = 1), np.percentile(koopman_erros[i], 75, axis = 1), alpha = 0.25, linewidth = 0, color=cmap(i))
        plt.xlabel('Time step')
        plt.ylabel('Orientation error wrt goal')
        legend = plt.legend()
        frame = legend.get_frame()
        frame.set_facecolor('0.9')
        frame.set_edgecolor('0.9')
        plt.savefig(fig_path_simu_goal_demo)
        plt.show()
        print("Finish the evaluation using different koopman dynamics!")
        sys.exit()

if __name__ == "__main__":
    path_to_file = sys.argv[1]
    plot_curves(path_to_file)