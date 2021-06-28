import subprocess
import statistics
runs = 1
eps = 20
alphas = [0.05, 0.1, 0.15, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
gammas = [0.99, 0.95, 0.9, 0.8]
resultados = {}
for alpha in alphas:
    resultados[alpha] = {}
    for gamma in gammas:
        process = subprocess.Popen(["python3", "experiments/ql_diamond.py", "-s", "5000", "-a", str(alpha), "-g", str(gamma), "-runs", str(runs), "-eps", str(eps)], stdout=subprocess.PIPE)
        stdout = process.communicate()[0]
        values = str(stdout).split('\\n')[-2].replace("[", "").replace("]", "").replace(" ", "").replace('\'', '').split(",")
        values = list(map(float, values))
        resultados[alpha][gamma] = {'values': values, 'sum': sum(values), 'mean': statistics.mean(values), 'std': statistics.pstdev(values)}

print("ql_diamond", resultados)
        # input()
