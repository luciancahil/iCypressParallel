file = open("./run/results/MARKER_COAD_ESET_grid_MARKER_COAD_ESET/agg/val.csv")
output = open("COAD-Rankings.csv", 'w')

lines = file.readlines()

def compare(line1):
    return line1.split(',')[-2]

lines = sorted(lines, key=compare, reverse=True)


for i, line in enumerate(lines):
    output.write("Rank {}: {}".format(i, line))