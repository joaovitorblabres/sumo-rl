import os
import numpy as np
import pandas as pd
import argparse
import glob
import statistics
from sklearn.preprocessing import MaxAbsScaler as scaler
import matplotlib.pyplot as plt

norm = None
def normalizer() -> scaler:
    global norm
    if norm is None:
        # path = "./outputs/BASELINE/alpha0.3_gamma0.8_eps0.05_decay1.0/2021-11-08/09:05:03/"
        path = "./outputs/FIXED/gamma0.8_eps0.05_decay1.0/2021-12-16/18:44:31/"
        fit_file = f"{path}fit_data.csv"
        # print(fit_file)
        try:
            fit_data = pd.read_csv(fit_file).to_numpy()
            norm = scaler().fit(fit_data)
        except FileNotFoundError:
            err_str = "Fit data must be in scenario directory."
            err_str += " Please run simulation with flag '--collect' before"
            raise RuntimeError(err_str) from FileNotFoundError
    return norm

prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                              description="""Plot Traffic Signal Metrics""")
prs.add_argument('-f', nargs='+', required=True, help="Measures files\n")
args = prs.parse_args()

for file in args.f:
    # print(sorted(glob.glob(file+'_r*'), key=os.path.getmtime))
    # files = sorted(glob.glob(file+'LC_*'))
    # ------ Normal -------
    # files = sorted(glob.glob(file+'_r*'))
    # runs = {saida.split('_')[-2] for saida in files}
    # ------ Density -------
    files = sorted(glob.glob(file+'_*_densities.csv'))
    runs = {saida.split('_')[-3] for saida in files}
    # print(files)
    # print(runs); exit()
    for run in runs:
        main_df = pd.DataFrame()
        mean_eps = []
        print(run)
        # exit()
        # for f in sorted(glob.glob(file+'LC_B5_'+run+'_*'), key=os.path.getmtime):
        for f in sorted(glob.glob(file+'_'+run+'_*'), key=os.path.getmtime):
        # for f in sorted(glob.glob(file+'_'+run[-1]+'*'), key=os.path.getmtime):
            df = pd.read_csv(f, sep=',')
            # mean_ep = df.groupby('step_time').mean()['total_wait_time']
            # mean_eps.append(statistics.mean(mean_ep))
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df), ignore_index=True)

        # norm = normalizer()
        for i in main_df.index:
            main_df.at[i, 'step_time'] = i * 5
            # r = [main_df['average_wait_time'][i]/13, main_df['flow'][i]/13]
            # reward = norm.transform([np.array(r)])[0] if norm else np.array(r)
            # main_df.at[i, 'reward'] = reward[0]*0.5 + reward[1]*0.5 # m*linear[0] + (1-m)*linear[1]


        # mean_df = pd.DataFrame(mean_eps)
        # for i in mean_df.index:
        #     mean_df.at[i, 'day'] = i
        # _ = plt.hist(main_df['average_wait_time'], bins=30, rwidth=1, edgecolor='black', linewidth=1.2)
        # plt.savefig(args.f[0]+'histogram_average_mod3.jpg', bbox_inches="tight")
        # hist, bin_edges = np.histogram(main_df['average_wait_time'], bins='auto')
        # hist, bin_edges = np.histogram(main_df['average_wait_time'], bins=100)
        # hist, bin_edges = np.histogram(main_df['average_wait_time'], bins='stone')
        # hist, bin_edges = np.histogram(main_df['average_wait_time'], bins=[0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 100, 150, 200, 300, 5000])
        # X = np.array([[main_df['flow'][i], main_df['average_wait_time'][i]] for i in range(0, len(main_df['flow']))])
        # fit = pd.DataFrame(X)
        # fit.to_csv(args.f[0]+"fit_data.csv", index=False)
        # transformer = MaxAbsScaler().fit(X)
        # print(hist, bin_edges, bin_edges.size, X[8000], transformer.transform(X)[8000])

        # main_df.to_csv(args.f[0]+"MLC_B5_"+run+".csv", index=False)
        # main_df.to_csv(args.f[0]+run+"_den_mean.csv", index=False)
        main_df.to_csv(args.f[0]+run+"_density_merged.csv", index=False)
