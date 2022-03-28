import statistics
import copy
import pandas as pd
import argparse
import glob
from matplotlib import pyplot as plt
import os
import numpy as np

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

def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

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
        gamma = params[2].split('/')[-1][5:]
        alpha = 0
        alphaG = 0
        # alphaG = params[3][6:]
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
                for f in sorted(glob.glob(hora+"/_r*"), key=os.path.getmtime):
                    df = pd.read_csv(f, sep=',')

                    if main_df.empty:
                        main_df = df
                    else:
                        main_df = pd.concat((main_df, df))

                    avg = df.groupby('step_time').sum()['total_wait_time']*-1
                    results[alpha][alphaG][gamma][gammaG][eps]['avg']['values'].append(sum(avg)/len(avg))
                    results[alpha][alphaG][gamma][gammaG][eps]['avg']['avgs'].append(statistics.mean(avg))
                    flow = df.groupby('step_time').sum()['flow']
                    results[alpha][alphaG][gamma][gammaG][eps]['flow']['values'].append(sum(flow)/len(flow))
                    results[alpha][alphaG][gamma][gammaG][eps]['flow']['avgs'].append(statistics.mean(flow))

                results[alpha][alphaG][gamma][gammaG][eps]['avg']['mean'].append(statistics.mean(results[alpha][alphaG][gamma][gammaG][eps]['avg']['values']))
                results[alpha][alphaG][gamma][gammaG][eps]['avg']['sum'].append(sum(results[alpha][alphaG][gamma][gammaG][eps]['avg']['values']))
                results[alpha][alphaG][gamma][gammaG][eps]['flow']['mean'].append(statistics.mean(results[alpha][alphaG][gamma][gammaG][eps]['flow']['values']))
                results[alpha][alphaG][gamma][gammaG][eps]['flow']['sum'].append(sum(results[alpha][alphaG][gamma][gammaG][eps]['flow']['values']))

    means = []
    sums = []
    gammas = []
    groups = []
    for g in results[alpha][alphaG].keys():
        gammas.append(g)
        groups.append(alpha)
        means.append(results[alpha][alphaG][g][gammaG][eps]['avg']['mean'] + results[alpha][alphaG][g][gammaG][eps]['flow']['mean'])
        sums.append(results[alpha][alphaG][g][gammaG][eps]['avg']['sum'] + results[alpha][alphaG][g][gammaG][eps]['flow']['sum'])

    par = paretoEfficient(np.array(means), False, True, False)
    mean_p = [gammas[p] for p in par]
    mean_g = [groups[p] for p in par]
    print(mean_p, mean_g, par, [means[p] for p in par])
    par = paretoEfficient(np.array(sums), False, True, False)
    sum_p = [gammas[p] for p in par]
    sum_g = [groups[p] for p in par]
    print(sum_p, sum_g, par, [sums[p] for p in par])
