import os
import numpy as np
import pandas as pd
import argparse
import glob

prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                              description="""Plot Traffic Signal Metrics""")
prs.add_argument('-f', nargs='+', required=True, help="Measures files\n")
args = prs.parse_args()

for file in args.f:
    main_df = pd.DataFrame()
    # print(sorted(glob.glob(file+'_r*'), key=os.path.getmtime))
    for f in sorted(glob.glob(file+'_r*'), key=os.path.getmtime):
        df = pd.read_csv(f, sep=',')
        if main_df.empty:
            main_df = df
        else:
            main_df = pd.concat((main_df, df), ignore_index=True)

for i in main_df.index:
    main_df.at[i, 'step_time'] = i * 5
# print(main_df)
main_df.to_csv(args.f[0]+"merged.csv", index=False)
