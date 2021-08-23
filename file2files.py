import os
import pandas as pd
import argparse

prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                              description="""Plot Traffic Signal Metrics""")
prs.add_argument('-f', nargs='+', required=True, help="Measures files\n")
args = prs.parse_args()

df = pd.read_csv(args.f[0], sep=',')

for label in df:
    if label == 'groups':
        valores = []
        for i, value in enumerate(df['groups']):
            groups = df['groups'][i].split('}, ')
            g = []
            for id, group in enumerate(groups):
                g.append(group.split(', Neighbours')[0][17:].strip().split(':')[1].strip())
            valores.append([int(df['step_time'][i])/5, len(g), g])

        newDF = pd.DataFrame(valores, columns =['step', 'number of groups', 'agents in the groups'])
        newDF.to_csv('groups_values.csv', index=False)
    # if label not in ['step_time', 'groups', 'recommendations'] and len(label) > 3:
    #     valores = []
    #     for i, value in enumerate(df[label]):
    #         v = eval(value)
    #         groups = df['groups'][i].split('}, ')
    #         gID = -1
    #         for id, group in enumerate(groups):
    #             if label[0:2] in group:
    #                 gID = id
    #                 continue
    #
    #         valores.append([int(df['step_time'][i])/5, v[0], v[1], v[4], v[3], gID])
    #         # print(label[0:2], valores, groups, value); exit()
    #     newDF = pd.DataFrame(valores, columns =['step', 'state', 'action', 'recommendation used?', 'reward', 'group id'])
    #     newDF.to_csv(label[0:2]+'_values.csv', index=False)
    print("TO CSV", label)
