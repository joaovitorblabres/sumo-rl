import os
import statistics
import copy
import numpy as np
import pandas as pd
import argparse
import glob
import matplotlib.pyplot as plt

def paretoEfficient(points, return_mask = True, repeated = False, minimize = True):
    """
    Find the (minimizing) pareto-efficient points
    :param points: An (n_points, n_points) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(points.shape[0])
    n_points = points.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(points):
        if minimize:
            nondominated_point_mask = np.any(points < points[next_point_index], axis=1)
        else:
            nondominated_point_mask = np.any(points > points[next_point_index], axis=1)
        if repeated:
            for i in range(points.shape[0]):
                if np.array_equal(points[next_point_index], points[i]):
                    nondominated_point_mask[i] = True
        else:
            nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        points = points[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask

    else:
        return is_efficient

def takeSecond(elem):
    return elem[1]

prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                              description="""Plot Traffic Signal Metrics""")
prs.add_argument('-f', nargs='+', required=True, help="Measures files\n")
args = prs.parse_args()
np.set_printoptions(suppress=True)

results = {}
for file in args.f:
    print(file)
    for alphas in sorted(glob.glob(file+"/*"), key=os.path.getmtime):
        main_df = pd.DataFrame()
        params = alphas.split('/')[1].split("_")
        gamma = params[0][5:]
        alpha = 0
        # alphaG = 0
        alphaG = params[1][2:]
        # gamma = 0
        # gamma = params[2][5:]
        gammaG = 0
        # gammaG = params[4][6:]
        eps = 0
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
            results[alpha][alphaG][gamma][gammaG][eps] = {'avg':{'sum':[],'values': [], 'mean':[], 'avgs': []},'flow':{'sum':[],'values': [], 'mean':[], 'avgs': []}}

        for data in glob.glob(alphas+"/*"):
            for hora in glob.glob(data+"/*"):
                print(hora)
                for f in sorted(glob.glob(hora+"/_r*"), key=lambda name: int(name.split('_')[-2][3:])):
                    df = pd.read_csv(f, sep=',')

                    if main_df.empty:
                        main_df = df
                    else:
                        main_df = pd.concat((main_df, df))

                    avg = df.groupby('step_time').sum()['average_wait_time']*-1
                    results[alpha][alphaG][gamma][gammaG][eps]['avg']['avgs'].append(statistics.mean(avg))
                    flow = df.groupby('step_time').sum()['flow']
                    results[alpha][alphaG][gamma][gammaG][eps]['flow']['avgs'].append(statistics.mean(flow))
                    run = int(f.split('_')[-2][3:])
                    # print(run, f)
                    if run == 1:
                        results[alpha][alphaG][gamma][gammaG][eps]['avg']['values'].append(sum(avg)/len(avg))
                        results[alpha][alphaG][gamma][gammaG][eps]['flow']['values'].append(sum(flow)/len(flow))
                    else:
                        pos = int(f.split('_')[-1].split('.')[0][2:])-1
                        results[alpha][alphaG][gamma][gammaG][eps]['avg']['values'][pos] += ((sum(avg)/len(avg)) - results[alpha][alphaG][gamma][gammaG][eps]['avg']['values'][pos])/run
                        results[alpha][alphaG][gamma][gammaG][eps]['flow']['values'][pos] += ((sum(flow)/len(flow)) - results[alpha][alphaG][gamma][gammaG][eps]['flow']['values'][pos])/run

                results[alpha][alphaG][gamma][gammaG][eps]['avg']['mean'].append(statistics.mean(results[alpha][alphaG][gamma][gammaG][eps]['avg']['values'][-20:]))
                results[alpha][alphaG][gamma][gammaG][eps]['avg']['sum'].append(sum(results[alpha][alphaG][gamma][gammaG][eps]['avg']['values']))
                results[alpha][alphaG][gamma][gammaG][eps]['flow']['mean'].append(statistics.mean(results[alpha][alphaG][gamma][gammaG][eps]['flow']['values'][-20:]))
                results[alpha][alphaG][gamma][gammaG][eps]['flow']['sum'].append(sum(results[alpha][alphaG][gamma][gammaG][eps]['flow']['values']))
        # print(results[alpha][alphaG][gamma][gammaG][eps]['avg']['mean'], results[alpha][alphaG][gamma][gammaG][eps]['flow']['mean'])

    means = []
    sums = []
    gammas = []
    groups = []
    for alphaG in results[alpha].keys():
        for g in results[alpha][alphaG].keys():
            gammas.append(g)
            groups.append(alphaG)
            means.append(results[alpha][alphaG][g][gammaG][eps]['avg']['mean'] + results[alpha][alphaG][g][gammaG][eps]['flow']['mean'])
            sums.append(results[alpha][alphaG][g][gammaG][eps]['avg']['sum'] + results[alpha][alphaG][g][gammaG][eps]['flow']['sum'])

    par = paretoEfficient(np.array(means), False, True, False)
    mean_p = [gammas[p] for p in par]
    mean_g = [groups[p] for p in par]
    print(mean_p, mean_g, par, [means[p] for p in par])
    # par = paretoEfficient(np.array(sums), False, True, False)
    # sum_p = [gammas[p] for p in par]
    # sum_g = [groups[p] for p in par]
    # print(sum_p, sum_g, par, [sums[p] for p in par])

flow = []
queue = []
for p in means:
    queue.append(p[0]*-1)
    flow.append(p[1])
plt.plot(flow, queue, 'o', color='#c3c3c3', markersize=6)

both = np.array(means)
f = []
q = []
pareto = [means[p] for p in par]
pareto = sorted(pareto, key=takeSecond)
print(pareto)
for p in pareto:
    q.append(p[0]*-1)
    f.append(p[1])
plt.ylabel("Tempo médio de espera")
# plt.xlabel("vehicles_on_network")
plt.xlabel("Veículos passando pela intersecção")
plt.plot(f, q, 'o-', color='black', markersize=6, linewidth=2)
plt.savefig(file+'PARETO.png')
