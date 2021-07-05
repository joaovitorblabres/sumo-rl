import subprocess
import statistics
runs = 1
eps = 20
alphas = [0.05, 0.1]
decays = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
alphasGroups = [0]
# alphasGroups = [0.05, 0.1, 0.15, 0.2]
gammas = [0.8, 0.85, 0.9, 0.95, 0.99, 0.9985]
gammasGroups = [0]
# gammasGroups = [0.8, 0.85, 0.9, 0.95, 0.99, 0.9985]
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
                    process = subprocess.Popen(["python3", "experiments/ql_diamond_withoutgroups.py", "-s", "20000", "-a", str(alpha), "-g", str(gamma), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps)], stdout=subprocess.PIPE)
                    # process = subprocess.Popen(["python3", "experiments/ql_diamond.py", "-s", "20000", "-a", str(alpha), "-g", str(gamma), "-ag", str(alphaG), "-gg", str(gammaG), "-e", str(decay), "-d", str(1), "-runs", str(runs), "-eps", str(eps)], stdout=subprocess.PIPE)
                    stdout = process.communicate()[0]
                    values = str(stdout).split('\\n')[-2].replace("[", "").replace("]", "").replace(" ", "").replace('\'', '').split(",")
                    values = list(map(float, values))
                    resultados[alpha][alphaG][gamma][gammaG][decay] = {'values': values, 'sum': sum(values), 'mean': statistics.mean(values), 'std': statistics.pstdev(values)}

print("ql_diamond_withoutgroups", resultados)
        # input()
