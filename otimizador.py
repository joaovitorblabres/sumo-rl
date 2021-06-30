import subprocess
import statistics
runs = 1
eps = 20
alphas = [0.05, 0.1]
alphasGroups = [0.05, 0.1, 0.2]
gammas = [0.9985, 0.99]
gammasGroups = [0.9985, 0.99, 0.9]
resultados = {}
for alpha in alphas:
    resultados[alpha] = {}
    for alphaG in alphasGroups:
        resultados[alpha][alphaG] = {}
        for gamma in gammas:
            resultados[alpha][alphaG][gamma] = {}
            for gammaG in gammasGroups:
                process = subprocess.Popen(["python3", "experiments/ql_diamond.py", "-s", "5000", "-a", str(alpha), "-g", str(gamma), "-ag", str(alphaG), "-gg", str(gammaG), "-runs", str(runs), "-eps", str(eps)], stdout=subprocess.PIPE)
                stdout = process.communicate()[0]
                values = str(stdout).split('\\n')[-2].replace("[", "").replace("]", "").replace(" ", "").replace('\'', '').split(",")
                values = list(map(float, values))
                resultados[alpha][alphaG][gamma][gammaG] = {'values': values, 'sum': sum(values), 'mean': statistics.mean(values), 'std': statistics.pstdev(values)}

print("ql_diamond", resultados)
        # input()
