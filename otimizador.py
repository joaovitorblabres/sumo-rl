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
                        for rec in results[alpha][alphaG][gamma][gammaG][decay].keys():
                            sums.append(results[alpha][alphaG][gamma][gammaG][decay][rec]['sum'])
                            mins.append(min(results[alpha][alphaG][gamma][gammaG][decay][rec]['values']))
                            means.append(results[alpha][alphaG][gamma][gammaG][decay][rec]['mean'])
                            moving.append(sum(results[alpha][alphaG][gamma][gammaG][decay][rec]['values'][-10:])/10)
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
eps = 100
# ['0.2-0-0.9-0-0.05', '0.3-0-0.8-0-0.05', '0.2-0-0.8-0-0.05', '0.1-0-0.95-0-0.05', '0.1-0-0.85-0-0.05', '0.1-0-0.9-0-0.05', '0.15-0-0.8-0-0.05', '0.15-0-0.9-0-0.05', '0.5-0-0.85-0-0.05', '0.1-0-0.8-0-0.05']
params = {}
# params[0] = {'alpha': 0.2, 'gamma': 0.9, 'epsilon': 0.05, 'decay': 1}
# params[1] = {'alpha': 0.3, 'gamma': 0.8, 'epsilon': 0.05, 'decay': 1}
# params[2] = {'alpha': 0.2, 'gamma': 0.8, 'epsilon': 0.05, 'decay': 1}
# params[3] = {'alpha': 0.1, 'gamma': 0.95, 'epsilon': 0.05, 'decay': 1}
# params[4] = {'alpha': 0.1, 'gamma': 0.85, 'epsilon': 0.05, 'decay': 1}
# params[5] = {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.05, 'decay': 1}
# params[6] = {'alpha': 0.15, 'gamma': 0.8, 'epsilon': 0.05, 'decay': 1}
# params[7] = {'alpha': 0.15, 'gamma': 0.9, 'epsilon': 0.05, 'decay': 1}
# params[8] = {'alpha': 0.5, 'gamma': 0.85, 'epsilon': 0.05, 'decay': 1}
# params[9] = {'alpha': 0.1, 'gamma': 0.8, 'epsilon': 0.05, 'decay': 1}
alphas = [0.0]
# alphas = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# alphas = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
decays = [0.05]
alphasGroups = [0]
# alphasGroups = [0.10, 0.15, 0.20, 0.3, 0.4, 0.5]
# gammas = [0.95]
# gammas = [0.001, 0.01, 0.05, 0.1]
# gammas = [0.2, 0.3, 0.4, 0.5]
# gammas = [0.6, 0.7, 0.80, 0.90]
# gammas = [0.95, 0.99, 0.995, 0.999]
# gammas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999]
gammas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999]
# gammas = [0.01]
# gammas = [0.80, 0.90, 0.95, 0.99, 0.995, 0.999]
gammasGroups = [0]
# gammasGroups = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.80, 0.90, 0.95, 0.99, 0.999]
# gammasGroups = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.80, 0.90, 0.95, 0.99, 0.999]
recs = [0.2]
resultados = {}
for alpha in alphas:
    resultados[alpha] = {}
    for alphaG in alphasGroups:
        resultados[alpha][alphaG] = {}
        for gamma in gammas:
            resultados[alpha][alphaG][gamma] = {}
            for gammaG in gammasGroups:
                resultados[alpha][alphaG][gamma][gammaG] = {}
                for decay in decays:
                    resultados[alpha][alphaG][gamma][gammaG][decay] = {}
                    for rec in recs:
                        print(gammaG, gamma)
                        # process = subprocess.Popen(["python3", "experiments/ql_diamond_withoutgroups.py", "-s", "20000", "-a", str(alpha), "-g", str(gamma), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps), "-optimize"], stdout=subprocess.PIPE)
                        # process = subprocess.Popen(["python3", "experiments/ql_diamond.py", "-s", "20000", "-a", str(alpha), "-g", str(gamma), "-ag", str(alphaG), "-gg", str(gammaG), "-e", str(decay), "-eg", str(rec), "-d", str(1), "-runs", str(runs), "-eps", str(eps)], stdout=subprocess.PIPE)
                        process = subprocess.Popen(["python3", "experiments/pql_diamond.py", "-s", "20000", "-g", str(gamma), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps), "-algType", str(1), "-optimize"], stdout=subprocess.PIPE)
                        # process = subprocess.Popen(["python3", "experiments/MOD_pql_diamond.py", "-s", "20000", "-g", str(gamma), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps), "-algType", str(0)], stdout=subprocess.PIPE)
                        # process = subprocess.Popen(["python3", "experiments/MOD_pql_diamond.py", "-s", "20000", "-g", str(gamma), "-gt", str(gammaG), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps), "-algType", str(1), "-optimize"], stdout=subprocess.PIPE)
                        stdout = process.communicate()[0]
                        # values = str(stdout).split('\\n')[-2].replace("[", "").replace("]", "").replace(" ", "").replace('\'', '').split(",")
                        # values = list(map(float, values))
                        # resultados[alpha][alphaG][gamma][gammaG][decay][rec] = {'values': values, 'sum': sum(values), 'mean': statistics.mean(values), 'std': statistics.pstdev(values)}
# resultados = {}
# for param in params:
#     alpha = params[param]['alpha']
#     gamma = params[param]['gamma']
#     decay = params[param]['epsilon']
#     resultados[alpha] = {}
#     resultados[alpha]['0'] = {}
#     resultados[alpha]['0'][gamma] = {}
#     resultados[alpha]['0'][gamma]['0'] = {}
#     process = subprocess.Popen(["python3", "experiments/ql_diamond_withoutgroups.py", "-s", "20000", "-a", str(alpha), "-g", str(gamma), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps)], stdout=subprocess.PIPE)
#     # process = subprocess.Popen(["python3", "experiments/ql_diamond.py", "-s", "20000", "-a", str(alpha), "-g", str(gamma), "-ag", str(alphaG), "-gg", str(gammaG), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps)], stdout=subprocess.PIPE)
#     stdout = process.communicate()[0]
#     values = str(stdout).split('\\n')[-2].replace("[", "").replace("]", "").replace(" ", "").replace('\'', '').split(",")
#     values = list(map(float, values))
#     resultados[alpha]['0'][gamma]['0'][decay] = {'values': values, 'sum': sum(values), 'mean': statistics.mean(values), 'std': statistics.pstdev(values)}

print(resultados)
bests(resultados)
        # input()
