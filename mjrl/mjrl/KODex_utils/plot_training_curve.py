"""
Plot the training curves using the records
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import re
import numpy as np

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
        with open(path_to_file) as f:
            lines = f.readlines()   
        loss_dict = {}
        loss_dict['total_loss'] = []
        loss_dict['encoder_loss'] = []
        loss_dict['pred_loss'] = []
        encoder_loss_visual = []
        pred_loss_visual = []
        for i in range(int(len(lines) / 3)):
            tmp_training = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", lines[3 * i + 1])
            loss_dict['total_loss'].append(float(tmp_training[1]))
            loss_dict['encoder_loss'].append(float(tmp_training[2]))
            loss_dict['pred_loss'].append(float(tmp_training[3]))
            tmp_visualization = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", lines[3 * i + 2])
            encoder_loss_visual.append(float(tmp_visualization[1])) 
            pred_loss_visual.append(float(tmp_visualization[2]))  
        plt.figure(1)
        plt.axes(frameon=0)
        plt.grid()
        plt.plot(loss_dict['total_loss'], linewidth=2, label = 'Total loss', color='#B22400')
        plt.plot(loss_dict['encoder_loss'], linewidth=2, label = 'Auto-encoder loss', color='#006BB2')
        plt.plot(loss_dict['pred_loss'], linewidth=2, label = 'Prediction loss', color='#F22BB2')
        plt.xlabel('Training iteration')
        plt.ylabel('Loss')
        legend = plt.legend()
        frame = legend.get_frame()
        frame.set_facecolor('0.9')
        frame.set_edgecolor('0.9')
        plt.figure(100)
        x = np.arange(0, len(encoder_loss_visual))
        plt.axes(frameon=0)
        plt.grid()
        plt.plot(x, encoder_loss_visual, linewidth=2, label = 'Encoder loss', color='#B22400')
        plt.plot(x, pred_loss_visual, linestyle='--', linewidth=2, label = 'Prediction loss', color='#006BB2')
        plt.xlabel('Training iteration')
        plt.ylabel('Prediction error of all traj data')
        legend = plt.legend()
        frame = legend.get_frame()
        frame.set_facecolor('0.9')
        frame.set_edgecolor('0.9')
        plt.show()

if __name__ == "__main__":
    path_to_file = sys.argv[1]
    plot_curves(path_to_file)