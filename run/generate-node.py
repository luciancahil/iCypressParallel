import os
import torch
import numpy as np
from NodeClassifyDataset import NodeClassifyDataset
import sys
from AnnDataProcessor import AnnDataProcessor
import yaml
import random
from sklearn.decomposition import PCA
from generation import make_grid, make_grid_sh, makeConfigFile, make_single_sh, make_replication

def process_nodes(node_file):
    file = open(node_file, mode="r")
    labels = []
    data = []

    for line in file:
        parts = line.split(",")
        labels.append(int(parts[0]))
        data.append([float(part) for part in parts[1:]])

    
    return data, labels

def process_edges(edge_file):
    start = []
    end = []
    file = open(edge_file, mode = "r")
    
    for line in file:
        parts = line.split(",")

        start.append(int(parts[0]))
        end.append(int(parts[1]))
    
    return [start, end]

def make_dataset(name, data, labels, edges):
    try:
        os.mkdir(os.path.join("datasets"))
    except OSError:
        pass

    try:
        os.mkdir(os.path.join("datasets", name))
    except OSError:
        pass

    try:
        os.mkdir(os.path.join("datasets", name ,"raw"))
    except OSError:
        pass

    try:
        os.mkdir(os.path.join("datasets", name, "processed"))
    except OSError:
        pass

    full_dataset = NodeClassifyDataset(root="data/", name=name, filename="full.csv", 
                                    test=True, x=data, y = labels, edge_label_index = edges)    
    
    torch.save(full_dataset, (os.path.join("datasets", name, "raw",  name + ".pt")))


#MAIN
# things we need to be provided

nodes = sys.argv[1]
nodes = os.path.join("rawData", nodes + ".csv")

edges = sys.argv[2]
edges = os.path.join("rawData", edges + ".csv")

name = sys.argv[3]

grid = sys.argv[4]
if grid[0] == "F":
    grid = False
else:
    grid = True



try:
    MAP_FILE = sys.argv[5]
    if(MAP_FILE.upper() == "NULL"):
        raise IndexError
    MAP_FILE = sys.argv[5]
except(IndexError):
    MAP_FILE = "GenesToTissues.csv"

try:
    if(sys.argv[6].upper() == "NULL"):
        raise IndexError
    parameter_file = sys.argv[6] + ".yaml"
except(IndexError):
    if(grid):
        parameter_file = "Default Grid.yaml"
    else:
        parameter_file = "Default Config.yaml"


try:
    num_genes = int(sys.argv[8][0])
except(IndexError):
    num_genes = 0

config_path = os.path.join("Hyperparameters", parameter_file)
with open(config_path, 'r') as file:
    configs = yaml.safe_load(file)

if(grid):
    single_config_path = os.path.join("Hyperparameters", "Default Config.yaml")
    with open(single_config_path, 'r') as file:
        single_configs = yaml.safe_load(file)


# general configs that we can keep as they are, unless changed.

#check to see if we have the csv or AnnData Files by looking at the first input
if sys.argv[1] == "ANN": # we have AnnData
    AnnData = True
    annPath = os.path.join("rawData", sys.argv[2]+".h5ad")
    adp = AnnDataProcessor(annPath)
    eset_name = sys.argv[1] + "-" + sys.argv[2]

else:
    AnnData = False
    eset_name = sys.argv[1]

data, labels = process_nodes(nodes)

edges = process_edges(edges)

make_dataset(name, data, labels, edges)


if (grid) :
    make_grid_sh(sys.argv[1], name, name)
    make_grid(name, configs, False)
    makeConfigFile(name, single_configs)
else:
    config_name = makeConfigFile(name, configs, False)
    make_single_sh(sys.argv[1], name, config_name)
#also need to make the grid file and the sh file