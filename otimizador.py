import subprocess
import statistics
import copy

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
                        # print(alpha, alphaG, gamma, gammaG, decay, results[alpha][alphaG][gamma][gammaG][decay]['mean'], results[alpha][alphaG][gamma][gammaG][decay]['sum'], results[alpha][alphaG][gamma][gammaG][decay]['values'][-10:], i)
                        i += 1

    original = copy.deepcopy(moving)
    print(min(moving), moving.index(min(moving)))
    moving.sort()
    print(moving[:10], [params[original.index(val)] for val in moving[:10]], [original.index(val) for val in moving[:10]])
    print(min(sums), sums.index(min(sums)))
    print(min(mins), mins.index(min(mins)))
    print(min(means), means.index(min(means)))

# alpha 0.14 - gamma: 0.94 - epsilon: 0.05
# alpha 0.14 - gamma: 0.91 - epsilon: 0.05
# alpha 0.11 - gamma: 0.86 - epsilon: 0.05
# alpha 0.13 - gamma: 0.95 - epsilon: 0.05
# alpha 0.15 - gamma: 0.95 - epsilon: 0.05
runs = 1
eps = 1
params = {}
params[0] = {'alpha': 0.14, 'gamma': 0.94, 'epsilon': 0.05, 'decay': 1}
params[1] = {'alpha': 0.14, 'gamma': 0.91, 'epsilon': 0.05, 'decay': 1}
params[2] = {'alpha': 0.11, 'gamma': 0.86, 'epsilon': 0.05, 'decay': 1}
params[3] = {'alpha': 0.13, 'gamma': 0.95, 'epsilon': 0.05, 'decay': 1}
params[4] = {'alpha': 0.15, 'gamma': 0.95, 'epsilon': 0.05, 'decay': 1}
# alphas = [0.14]
# decays = [0.05]
# alphasGroups = [0]
# # alphasGroups = [0.05, 0.1, 0.15, 0.2]
# gammas = [0.91, 0.94]
# gammasGroups = [0]
# gammasGroups = [0.8, 0.85, 0.9, 0.95, 0.99, 0.9985]
# resultados = {}
# for alpha in alphas:
#     resultados[alpha] = {}
#     for alphaG in alphasGroups:
#         resultados[alpha][alphaG] = {}
#         for gamma in gammas:
#             resultados[alpha][alphaG][gamma] = {}
#             for gammaG in gammasGroups:
#                 resultados[alpha][alphaG][gamma][gammaG] = {}
#                 for decay in decays:
                    # process = subprocess.Popen(["python3", "experiments/ql_diamond_withoutgroups.py", "-s", "20000", "-a", str(alpha), "-g", str(gamma), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps)], stdout=subprocess.PIPE)
                    # # process = subprocess.Popen(["python3", "experiments/ql_diamond.py", "-s", "20000", "-a", str(alpha), "-g", str(gamma), "-ag", str(alphaG), "-gg", str(gammaG), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps)], stdout=subprocess.PIPE)
                    # stdout = process.communicate()[0]
                    # values = str(stdout).split('\\n')[-2].replace("[", "").replace("]", "").replace(" ", "").replace('\'', '').split(",")
                    # values = list(map(float, values))
                    # resultados[alpha][alphaG][gamma][gammaG][decay] = {'values': values, 'sum': sum(values), 'mean': statistics.mean(values), 'std': statistics.pstdev(values)}
results = {}
for param in params:
    alpha = params[param]['alpha']
    gamma = params[param]['gamma']
    decay = params[param]['decay']
    results[alpha] = {}
    results[alpha]['0'] = {}
    results[alpha]['0'][gamma] = {}
    results[alpha]['0'][gamma]['0'] = {}
    process = subprocess.Popen(["python3", "experiments/ql_diamond_withoutgroups.py", "-s", "200", "-a", str(alpha), "-g", str(gamma), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps)], stdout=subprocess.PIPE)
    # process = subprocess.Popen(["python3", "experiments/ql_diamond.py", "-s", "20000", "-a", str(alpha), "-g", str(gamma), "-ag", str(alphaG), "-gg", str(gammaG), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps)], stdout=subprocess.PIPE)
    stdout = process.communicate()[0]
    values = str(stdout).split('\\n')[-2].replace("[", "").replace("]", "").replace(" ", "").replace('\'', '').split(",")
    values = list(map(float, values))
    results[alpha]['0'][gamma]['0'][decay] = {'values': values, 'sum': sum(values), 'mean': statistics.mean(values), 'std': statistics.pstdev(values)}

print(results)
bests(results)
        # input()
