import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                              description="""Plot Traffic Signal Metrics""")
prs.add_argument('-f', nargs='+', required=True, help="Measures files\n")
args = prs.parse_args()

df = pd.read_csv(args.f[0], sep=',')

valores = {}
val = {}
for i, value in enumerate(df['groups']):
    groups = df['groups'][i].split('}, ')
    for id, group in enumerate(groups):
        # print(i, valores, group, eval(group.split(', Reward:')[1].replace('}', '')))
        g = group.split(', Neighbours')[0][17:].strip().split(':')[1].strip()
        r = eval(group.split(', Reward:')[1].replace('}', ''))
        if g in valores:
            valores[g]['reward'] += r
            val[g].append(r)
            valores[g]['times'] += 1
        else:
            valores[g] = {'reward': r, 'times': 0, 'mean': 0, 'std': 0}
            val[g] = [r]

    # print(i, valores)
# exit()

newDF = pd.DataFrame.from_dict(valores)
for g in val:
    mean = np.mean(val[g])
    std = np.std(val[g])
    newDF[g]['mean'] = mean
    newDF[g]['std'] = std
newDF.to_csv('groups.csv')
