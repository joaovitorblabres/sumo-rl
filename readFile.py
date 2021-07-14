import ast
import copy

# f = open("resultsGroupsDecayFixed.txt", "r")
f = open("saidaBigONE.txt", "r")
results = ast.literal_eval(f.read())
sums = []
means = []
mins = []
moving = []
params = {}
i = 0
for alpha in results.keys():
    for alphaG in results[alpha].keys():
        for gamma in results[alpha][alphaG].keys():
            # for gammaG in results[alpha][alphaG][gamma].keys():
                sums.append(results[alpha][alphaG][gamma]['sum'])
                mins.append(min(results[alpha][alphaG][gamma]['values']))
                means.append(results[alpha][alphaG][gamma]['mean'])
                moving.append(sum(results[alpha][alphaG][gamma]['values'][-10:])/10)
                params[i] = "-".join([str(alpha), str(alphaG), str(gamma)])
                print(alpha, alphaG, gamma, results[alpha][alphaG][gamma], i)
                i += 1

original = copy.deepcopy(moving)
print(min(moving), moving.index(min(moving)))
moving.sort()
print(moving[:10], [params[original.index(val)] for val in moving[:10]], [original.index(val) for val in moving[:10]])
print(min(sums), sums.index(min(sums)))
print(min(mins), mins.index(min(mins)))
print(min(means), means.index(min(means)))
