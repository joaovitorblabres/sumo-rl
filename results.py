import statistics
import copy
import pandas as pd
import argparse
import glob
import os

def bests(results):
    sums = []
    means = []
    mins = []
    moving = []
    params = {}
    i = 0
    for alpha in results.keys():
        for alphaG in results[alpha].keys():
            for gamma in results[alpha][alphaG].keys():
                for gammaG in results[alpha][alphaG][gamma].keys():
                    for decay in results[alpha][alphaG][gamma][gammaG].keys():
                        sums.append(results[alpha][alphaG][gamma][gammaG][decay]['sum'])
                        mins.append(min(results[alpha][alphaG][gamma][gammaG][decay]['values']))
                        means.append(results[alpha][alphaG][gamma][gammaG][decay]['mean'])
                        moving.append(sum(results[alpha][alphaG][gamma][gammaG][decay]['values'][-10:])/10)
                        params[i] = "-".join([str(alpha), str(alphaG), str(gamma), str(gammaG), str(decay)])
                        # print(alpha, alphaG, gamma, gammaG, decay, results[alpha][alphaG][gamma][gammaG][decay]['mean'], results[alpha][alphaG][gamma][gammaG][decay]['sum'], results[alpha][alphaG][gamma][gammaG][decay]['values'][-20:], i)
                        i += 1

    original = copy.deepcopy(moving)
    print(min(moving), moving.index(min(moving)))
    moving.sort()
    print(moving[:10], [params[original.index(val)] for val in moving[:10]], [original.index(val) for val in moving[:10]])
    print(min(sums), params[sums.index(min(sums))])
    print(min(mins), params[mins.index(min(mins))])
    print(min(means), params[means.index(min(means))])

prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                              description="""Plot Traffic Signal Metrics""")
prs.add_argument('-f', nargs='+', required=True, help="Measures files\n")
args = prs.parse_args()

results = {}
for file in args.f:
    print(file)
    for alphas in glob.glob(file+"/*"):
        main_df = pd.DataFrame()
        params = alphas.split("_")
        alpha = params[0].split('/')[-1][5:]
        gamma = params[1][5:]
        eps = params[2][3:]
        if alpha not in results.keys():
            results[alpha] = {}
            results[alpha][0] = {}
            if gamma not in results[alpha][0].keys():
                results[alpha][0][gamma] = {}
                results[alpha][0][gamma][0] = {}
                if eps not in results[alpha][0][gamma][0].keys():
                    results[alpha][0][gamma][0][eps] = {'sum':[],'values': [], 'mean':[]}
        if gamma not in results[alpha][0].keys():
            results[alpha][0][gamma] = {}
            results[alpha][0][gamma][0] = {}
            if eps not in results[alpha][0][gamma][0].keys():
                results[alpha][0][gamma][0][eps] = {'sum':[],'values': [], 'mean':[]}
        if eps not in results[alpha][0][gamma][0].keys():
            results[alpha][0][gamma][0][eps] = {'sum':[],'values': [], 'mean':[]}

        for data in glob.glob(alphas+"/*"):
            for hora in glob.glob(data+"/*"):
                for f in sorted(glob.glob(hora+"/_r*"), key=os.path.getmtime):
                    df = pd.read_csv(f, sep=',')

                    if main_df.empty:
                        main_df = df
                    else:
                        main_df = pd.concat((main_df, df))

                    all = df.groupby('step_time').sum()['total_wait_time']
                    results[alpha][0][gamma][0][eps]['values'].append(sum(all))
                results[alpha][0][gamma][0][eps]['mean'].append(statistics.mean(results[alpha][0][gamma][0][eps]['values']))
                results[alpha][0][gamma][0][eps]['sum'].append(sum(results[alpha][0][gamma][0][eps]['values']))

bests(results)
