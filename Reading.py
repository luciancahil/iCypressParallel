name = "THCA"

file = open("./run/results/MARKER_{}_ESET_grid_MARKER_{}_ESET/agg/val.csv".format(name, name))
output = open("{}-Rankings.csv".format(name), 'w')
single_grid = open("./run/Hyperparameters/{}-grid.yaml".format(name), 'w')

lines = file.readlines()

def compare(line1):
    return line1.split(',')[-2]

lines = sorted(lines, key=compare, reverse=True)


for i, line in enumerate(lines):
    output.write("Rank {}: {}".format(i, line))


output.close()

print(lines[1])

best = lines[1].split(",")

l_pre = "l_pre : [{}]\n".format(best[6])
l_mp = "l_mp : [{}]\n".format(best[7])
l_post = "l_post : [{}]\n".format(best[8])
stage = "stage : ['{}']\n".format(best[9])
agg = "agg : ['{}']\n".format(best[10])

single_grid.write(l_pre)
single_grid.write(l_mp)
single_grid.write(l_post)
single_grid.write(stage)
single_grid.write(agg)

print(lines[1])