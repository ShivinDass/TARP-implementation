from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import get_monitor_files
import numpy as np
import csv
import json
import pandas

def load_results(path: str) -> pandas.DataFrame:
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``

    :param path: the directory path containing the log file(s)
    :return: the logged data
    """
    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        print("ERROR")
        exit(0)
    data_frames, headers = [], []
    for file_name in monitor_files:
        with open(file_name, "rt") as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            data_frame = pandas.read_csv(file_handler, index_col=None)
            headers.append(header)
            data_frame["t"] += header["t_start"]
        data_frames.append(data_frame)
    return data_frames


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    data_frames = load_results(log_folder)#ts2xy(load_results(log_folder), 'timesteps'))
    # print(data_frames[0], data_frames[1])
    fig = plt.figure(title+" Smoothed")
    for i in range(len(data_frames)):
        x, y = ts2xy(data_frames[i], 'timesteps')
        y = moving_average(y, window=500)
        # Truncate x
        x = x[len(x) - len(y):]

        plt.plot(x, y, label = labels[i])
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.legend()
    plt.show()


labels = ["Reward Prediction Finetune", "Image Scratch", "Oracle"]
log_dir = "logs/cur/"
# results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
# plt.show()
plot_results(log_dir)