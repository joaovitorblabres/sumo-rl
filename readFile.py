import ast

# f = open("resultsGroupsDecayFixed.txt", "r")
f = open("resultsSolosDecayFixed2.txt", "r")
results = ast.literal_eval(f.read())
sums = []
means = []
mins = []
i = 0
for alpha in results.keys():
    for alphaG in results[alpha].keys():
        # for gamma in results[alpha][alphaG].keys():
            # for gammaG in results[alpha][alphaG][gamma].keys():
                sums.append(results[alpha][alphaG]['sum'])
                mins.append(min(results[alpha][alphaG]['values']))
                means.append(results[alpha][alphaG]['mean'])
                print(alpha, alphaG, results[alpha][alphaG], i)
                i += 1

print(min(sums), sums.index(min(sums)))
print(min(mins), mins.index(min(mins)))
print(min(means), means.index(min(means)))
