import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, ListedColormap, BoundaryNorm
import numpy as np

def plot_result(real, predict, class_predict=None, zero_line=True):
    plt.rcParams.update({'figure.figsize':(15,8), })
    plt.plot(list(real), label="real", marker="o", color="#ff7f0e")
    plt.plot(list(predict), label="predict", marker="o", color="#1f77b4")
    if zero_line:
        plt.plot([0 for i in range(len(real))], color='k')
    if type(class_predict) != type(None):
        plt.plot([0.1 if c == 1 else -0.1 for c in class_predict], label="class_predict", marker="o")
    plt.legend()
    plt.show()

def plot_diff_scale_plot(p1, p2, p1_name, p2_name):
    fig, axs = plt.subplots(1, 1, figsize=(15,8), sharex=True)
    
    fig.subplots_adjust(hspace=0)

    color1 = 'tab:blue'
    axs.set_ylabel(p1_name, color=color1)
    axs.plot(p1, color=color1) 
    
    color2 = 'tab:orange'
    ax2 = axs.twinx()
    ax2.set_ylabel(p2_name, color=color2)
    ax2.plot(p2, color=color2)

def plot_line(series, zero_line=True):
    plt.rcParams.update({'figure.figsize':(15,8), })
    plt.plot(list(series), marker="o", color="#1f77b4")
    if zero_line:
        plt.plot([0 for i in range(len(series))], color='k')
    plt.show()

def plot_state(data, feature, state_col, save=False, save_path=""):
    # fig, ax = plt.subplots(figsize=(20,6))
    n_state = len(data[state_col].unique())
    colors=["red", "green", "blue", "orange", "gold", "limegreen", "k",  "#550011", "purple", "seagreen"]
    fig, ax1 = plt.subplots(figsize=(20,8))
    # ax.plot(data[feature])
    cmap = ListedColormap(colors, 'indexed')
    norm = BoundaryNorm(range(n_state + 1), cmap.N)
    inxval = mdates.date2num(data.index.to_pydatetime())
    points = np.array([inxval, data[feature]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(data[state_col])
    plt.gca().add_collection(lc)
    plt.xlim(data.index.min(), data.index.max())
    plt.ylim(data[feature].min(), data[feature].max())
    color_patchs = []
    for i in range(n_state):
        color_patchs.append(mpatches.Patch(color=colors[i], label=str(i)))
    plt.legend(handles=color_patchs)
    if save:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_hist(series, bin_size=1, bin_range=None):
    if not bin_range:
        bin_range = (int(min(series)), int(max(series)))
    plt.rcParams.update({'figure.figsize':(15,8), })
    plt.hist(series, bins=range(bin_range[0], bin_range[1] + 1, bin_size))
    plt.show()

def plot_multiple_hist(df, bin_size=1, bin_range=None):
    if not bin_range:
        bin_range = (int(min(df)), int(max(df)))
    plt.rcParams.update({'figure.figsize':(15,8), })
    for column in df.columns:
        plt.hist(df[column], range(bin_range[0], bin_range[1] + 1, bin_size), alpha=0.5, label=column)
    plt.xticks(range(bin_range[0], bin_range[1] + 1, bin_size))
    plt.legend(loc='upper right')
    plt.show()