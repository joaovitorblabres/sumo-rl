import os
import numpy as np
import pandas as pd
import argparse
import glob
import statistics

prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                              description="""Plot Traffic Signal Metrics""")
prs.add_argument('-f', nargs='+', required=True, help="Measures files\n")
args = prs.parse_args()

for file in args.f:
    # print(sorted(glob.glob(file+'_r*'), key=os.path.getmtime))
    files = sorted(glob.glob(file+'_r*'))
    runs = {saida.split('_')[-2] for saida in files}
    for run in runs:
        main_df = pd.DataFrame()
        mean_eps = []
        print(run)
        for f in sorted(glob.glob(file+'_densities_'+run+'_*'), key=os.path.getmtime):
            df = pd.read_csv(f, sep=',')
            # mean_ep = df.groupby('step_time').mean()['total_wait_time']
            # mean_eps.append(statistics.mean(mean_ep))
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df), ignore_index=True)

        for i in main_df.index:
            main_df.at[i, 'step_time'] = i * 5

        mean_df = pd.DataFrame(mean_eps)
        for i in mean_df.index:
            mean_df.at[i, 'day'] = i

        # mean_df.to_csv(args.f[0]+run+"_den_mean.csv")
        main_df.to_csv(args.f[0]+run+"_den_merged.csv", index=False)
