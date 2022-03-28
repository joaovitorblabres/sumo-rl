import statistics
import copy
import pandas as pd
import argparse
import glob
from matplotlib import pyplot as plt
import os
import numpy as np

def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def bests(results):
    sums = []
    means = []
    mins = []
    moving = []
    avg = []
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
                        avg.append(results[alpha][alphaG][gamma][gammaG][decay]['avgs'])
                        moving.append(sum(results[alpha][alphaG][gamma][gammaG][decay]['values'][-10:])/10)
                        params[i] = "-".join([str(alpha), str(alphaG), str(gamma), str(gammaG), str(decay)])
                        # print(alpha, alphaG, gamma, gammaG, decay, results[alpha][alphaG][gamma][gammaG][decay]['mean'], results[alpha][alphaG][gamma][gammaG][decay]['sum'], results[alpha][alphaG][gamma][gammaG][decay]['values'][-20:], i)
                        i += 1

    original = copy.deepcopy(moving)
    print(min(moving), moving.index(min(moving)))
    moving.sort()
    print(moving[:10], [params[original.index(val)] for val in moving[:10]], [original.index(val) for val in moving[:10]])
    print(min(sums), params[sums.index(min(sums))])
    # print(avg)
    print(min(mins), params[mins.index(min(mins))])
    print(min(means), params[means.index(min(means))])
    prg = copy.deepcopy(means)
    means.sort()
    print(means[:10], [params[prg.index(val)] for val in means[:10]])

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
        alpha = params[2].split('/')[-1][5:]
        # alpha = params[1].split('/')[-1][5:]
        alphaG = 0
        # alphaG = params[3][6:]
        gamma = params[3][5:]
        # gamma = params[2][5:]
        gammaG = 0
        # gammaG = params[4][6:]
        eps = params[4][3:]
        # eps = params[5][3:]
        print(alpha, gamma, eps, alphaG, gammaG)
        if alpha not in results.keys():
            results[alpha] = {}
        if alphaG not in results[alpha].keys():
            results[alpha][alphaG] = {}
        if gamma not in results[alpha][alphaG].keys():
            results[alpha][alphaG][gamma] = {}
        if gammaG not in results[alpha][alphaG][gamma].keys():
            results[alpha][alphaG][gamma][gammaG] = {}
        if eps not in results[alpha][alphaG][gamma][gammaG].keys():
            results[alpha][alphaG][gamma][gammaG][eps] = {'sum':[],'values': [], 'mean':[], 'avgs': []}

        for data in glob.glob(alphas+"/*"):
            for hora in glob.glob(data+"/*"):
                print(hora)
                for f in sorted(glob.glob(hora+"/_r*"), key=os.path.getmtime):
                    # print(f)
                    df = pd.read_csv(f, sep=',')

                    # if main_df.empty:
                        # main_df = df
                    # else:
                        # main_df = pd.concat((main_df, df))

                    all = df.groupby('step_time').sum()['total_wait_time']
                    # all = df.groupby('step_time').sum()['flow']*-1
                    if int(f.split('_')[-2][3:]) == 1:
                        results[alpha][alphaG][gamma][gammaG][eps]['values'].append(sum(all)/len(all))
                        # print(f.split('_')[-1].split('.')[0][2:], results[alpha][alphaG][gamma][gammaG][eps]['values'][int(f.split('_')[-1].split('.')[0][2:])-1], (sum(all)/len(all)))
                    else:
                        # print(f.split('_')[-1].split('.')[0][2:], results[alpha][alphaG][gamma][gammaG][eps]['values'][int(f.split('_')[-1].split('.')[0][2:])-1], (sum(all)/len(all)))
                        results[alpha][alphaG][gamma][gammaG][eps]['values'][int(f.split('_')[-1].split('.')[0][2:])-1] += ((sum(all)/len(all)) - results[alpha][alphaG][gamma][gammaG][eps]['values'][int(f.split('_')[-1].split('.')[0][2:])-1])/int(f.split('_')[-2][3:])
                    # print(f.split('_')[-1].split('.')[0][2:], results[alpha][alphaG][gamma][gammaG][eps]['values'][int(f.split('_')[-1].split('.')[0][2:])-1], (sum(all)/len(all)))

                    # print(results[alpha][alphaG][gamma][gammaG][eps]['values'])
                    results[alpha][alphaG][gamma][gammaG][eps]['avgs'].append(statistics.mean(all))
                # exit()
                # print(sum(results[alpha][alphaG][gamma][gammaG][eps]['values'][-20:])/20)
                results[alpha][alphaG][gamma][gammaG][eps]['mean'].append(statistics.mean(results[alpha][alphaG][gamma][gammaG][eps]['values']))
                results[alpha][alphaG][gamma][gammaG][eps]['sum'].append(sum(results[alpha][alphaG][gamma][gammaG][eps]['values']))
                # if any([val < 10000 for val in results[alpha][alphaG][gamma][gammaG][eps]['avgs']]):
                # meanDF = pd.DataFrame(results[alpha][alphaG][gamma][gammaG][eps]['avgs'][:])
                # plt.xlim([-10,910])
                # plt.plot(range(0,len(results[alpha][alphaG][gamma][gammaG][eps]['avgs'][:])), moving_average(results[alpha][alphaG][gamma][gammaG][eps]['avgs'][:], 1), 'ro-')
                # for i in range(0, 3):
                    # plt.axvspan((3*i)*100, (1+(3*i))*100-1, facecolor='lightblue', alpha=0.5)
                    # plt.axvspan((1+(3*i))*100, (2+(3*i))*100-1, facecolor='coral', alpha=0.5)
                    # plt.axvspan((2+(3*i))*100, (3+(3*i))*100-1, facecolor='lawngreen', alpha=0.5)
                # plt.vlines([100-1, 200-1, 300-1, 400-1, 500-1, 600-1, 700-1, 800-1], -1000, min(results[alpha][alphaG][gamma][gammaG][eps]['avgs'])+1000)
                # plt.xlabel("Simulated Days")
                # plt.ylabel("Average Waited Time")
                # plt.title(f)
                # plt.show()
                    # yes = input()
                    # if yes == 's':
                # meanDF.to_csv(hora+"merged.csv")
                # results[alpha][alphaG][gamma][gammaG][eps] = {'sum':[],'values': [], 'mean':[], 'avgs': []}

bests(results)
