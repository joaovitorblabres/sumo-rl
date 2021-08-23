import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import glob
from itertools import cycle

sns.set(style='darkgrid', rc={'figure.figsize': (7.2, 4.45),
                            # 'text.usetex': True,
                            'xtick.labelsize': 25,
                            'ytick.labelsize': 25,
                            'font.size': 15,
                            'figure.autolayout': True,
                            'axes.titlesize' : 16,
                            'axes.labelsize' : 25,
                            'lines.linewidth' : 2,
                            'lines.markersize' : 6,
                            'legend.fontsize': 15})
# colors = sns.color_palette("dark")
#colors = sns.color_palette("Set1", 2)
colors = ['darkred','black','#3465a4', 'b', 'b', '#6a3d9a','#fb9a99']
# colors = ['#FF4500','#e31a1c','#329932', 'b', 'b', '#6a3d9a','#fb9a99']
dashes_styles = cycle(['-', '-', '-', ':'])
markers = cycle(['o', 'v', '*'])
sns.set_palette(colors)
colors = cycle(colors)

def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
pd.set_option("display.max_rows", None, "display.max_columns", None)

def plot_df(df, color, xaxis, yaxis, ma=1, label=''):
    df[yaxis] = pd.to_numeric(df[yaxis], errors='coerce')  # convert NaN string to NaN value
    # plt.ylim(0, 100000)
    mean = []
    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]
    # std = 0
    if ma > 1:
        mean = moving_average(mean, ma)
        std = moving_average(std, ma)

    x = df.groupby(xaxis)[xaxis].mean().keys().values
    # plt.plot(x, mean, label=label, color=color, marker=next(markers))
    # pd.DataFrame(mean).to_csv(args.f[0]+"ma.csv")
    # plt.legend(label)
    plt.plot(x, mean, label=label, color=color, linestyle=next(dashes_styles))
    print(max(mean), mean[45000:75000].mean(), mean[45000:75000].std())
    # meanDF = pd.DataFrame(mean)
    # meanDF.to_csv(label+"mean.csv")
    # plt.plot(x, df.groupby(xaxis).mean()['vehicles'], label='vehicles')
    # plt.fill_between(x, mean, mean, alpha=0.25, color=color, rasterized=True)
    plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)

    plt.ylim([0, 170000])
    plt.xlim([0, 75000])


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Plot Traffic Signal Metrics""")
    prs.add_argument('-f', nargs='+', required=True, help="Measures files\n")
    prs.add_argument('-l', nargs='+', default=None, help="File's legends\n")
    prs.add_argument('-t', type=str, default="", help="Plot title\n")
    prs.add_argument("-yaxis", type=str, default='total_wait_time', help="The column to plot.\n")
    prs.add_argument("-xaxis", type=str, default='step_time', help="The x axis.\n")
    prs.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
    prs.add_argument('-sep', type=str, default=',', help="Values separator on file.\n")
    prs.add_argument('-xlabel', type=str, default='Second', help="X axis label.\n")
    prs.add_argument('-ylabel', type=str, default='Total waiting time (s)', help="Y axis label.\n")
    prs.add_argument('-output', type=str, default=None, help="PDF output filename.\n")

    args = prs.parse_args()
    labels = cycle(args.l) if args.l is not None else cycle([str(i) for i in range(len(args.f))])
    # print(next(labels))
    plt.figure()

    # File reading and grouping
    for file in args.f:
        main_df = pd.DataFrame()
        # print(file, glob.glob(file+'*.csv'))
        # for f in glob.glob(file+'_r*'):
        for f in glob.glob(file+'*merged.csv'):
            df = pd.read_csv(f, sep=args.sep)
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))
            # print(main_df, f)

        # Plot DataFrame
        plot_df(main_df,
                xaxis=args.xaxis,
                yaxis=args.yaxis,
                label=next(labels),
                color=next(colors),
                ma=args.ma)
        del main_df

    plt.title(args.t)
    plt.ylabel(args.ylabel)
    plt.xlabel(args.xlabel)
    plt.legend(prop={'size': 35})
    # plt.rc('legend', fontsize=50)
    plt.ylim(bottom=0)
    # main_df.to_csv(args.f[0]+"merged.csv")

    if args.output is not None:
        plt.savefig(args.output+'.pdf', bbox_inches="tight")

    plt.show()
