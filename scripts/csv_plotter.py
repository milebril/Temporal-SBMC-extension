import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle
import os
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

file_location = '/home/emil//Documents/Temporal-SBMC-extension/output/emil/trained_models/final_v3/aa_csv_data/'

all = 'final_v2_all'
all_scratch = 'final_v2_all_scratch'
last = 'final_v2_last'
first = 'final_v2_first'
half = 'final_v2_half'
quarter = 'final_v2_quarter'
lowlr = 'final_v2_lowlr'

tags = ['Loss/train', 'RMSE/train', 'Loss/validation', 'RMSE/validation']

# def smooth(scalars, weight: float):  # Weight between 0 and 1
#     last = scalars[0]  # First value in the plot (first timestep)
#     smoothed = list()
#     for point in scalars:
#         smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
#         smoothed.append(smoothed_val)                        # Save it
#         last = smoothed_val                                  # Anchor the last smoothed value

#     return smoothed


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


from matplotlib.transforms import Bbox
def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

if __name__ == '__main__':
    plots = [all, all_scratch]
    labels = ["Loaded weights", "Scratch"]
    # tag = tags[2]
    colors = ['red', 'green', 'orange']

    save = False
    start = 5

    for i, tag in enumerate(tags):
        if save:
            fig, axeslist = plt.subplots(ncols=1, nrows=1)
            # fig.set_size_inches(8, 6)
            ax = axeslist
        else:
            fig, axeslist = plt.subplots(ncols=2, nrows=2)
            ax = axeslist.ravel()[i]
        ax.set_title(tag)
        
        for idx, plot in enumerate(plots):
            with open(file_location + plot + '.p', 'rb') as f:
                data = pickle.load(f)
                data = data[tag][start:]
                x_axis = np.arange(start, start + len(data))
                if 'train' not in tag:
                    ax.plot(x_axis, data, color=colors[idx], alpha=0.2)
                    smoothed_data = smooth(data, 0.9)
                    ax.plot(x_axis, smoothed_data, color=colors[idx], label=labels[idx])
                else:
                    ax.plot(x_axis, data, color=colors[idx], label=labels[idx])

        ax.legend()
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

        # extent = full_extent(ax)
        if save:
            fig.savefig(f'{file_location}{tag.replace("/", "_")}.png', bbox_inches= 'tight', dpi=100)

        # ax.savefig(f'{file_location}{tag}.png', bbox_inches="tight")
    plt.tight_layout(pad=3.0)
    plt.show()

