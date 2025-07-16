import os
import torch
import numpy as np
from CytokinesDataSet import CytokinesDataSet
import sys
from AnnDataProcessor import AnnDataProcessor
import yaml
import random
from sklearn.decomposition import PCA


# A file that is meant to generate all the files needed to run the underlying graphgym.


#TODO Edit this for more flexibility
def makeConfigFile(name, configs, isGraph = True):
    if (not os.path.exists(os.path.abspath("configs"))):
        os.makedirs(os.path.abspath("configs"))
    # using with statement
    with open(os.path.join('configs', name + ".yaml"), 'w') as file:
        file.write('out_dir: results\n')
        file.write('dataset:\n')
        file.write(' format: PyG\n')
        file.write(' name: Custom,' + name + ',,\n')
        if(isGraph):
            file.write(' task: graph\n')
        else:
            file.write(' task: node\n')
        file.write(' task_type: classification\n')
        file.write(' transductive: True\n')
        file.write(' split: [0.8, 0.2]\n')
        file.write(' augment_feature: []\n')
        file.write(' augment_feature_dims: [0]\n')
        file.write(' augment_feature_repr: position\n')
        file.write(' augment_label: \'\'\n')
        file.write(' augment_label_dims: 0\n')
        file.write(' transform: none\n')
        file.write('train:\n')
        file.write(' batch_size: ' + str(configs['batch_size']) + '\n' )
        file.write(' eval_period: ' + str(configs['eval_period']) + '\n')
        file.write(' ckpt_period: 100\n')
        file.write('model:\n')
        file.write(' type: gnn\n')
        file.write(' loss_fun: cross_entropy\n')
        file.write(' edge_decoding: dot\n')
        file.write(' graph_pooling: add\n')
        file.write('gnn:\n')
        file.write(' layers_pre_mp: ' + str(configs['layers_pre_mp']) + '\n')
        file.write(' layers_mp: ' + str(configs['layers_mp']) + '\n')
        file.write(' layers_post_mp: ' + str(configs['layers_post_mp']) + '\n')
        file.write(' dim_inner: ' + str(configs['dim_inner']) + '\n')
        file.write(' layer_type: gatconv\n')
        file.write(' stage_type: ' + str(configs['stage_type']) + ' \n')
        file.write(' batchnorm: True\n')
        file.write(' act: prelu\n')
        file.write(' dropout: 0.0\n')
        file.write(' agg: ' + configs['agg'] + '\n')
        file.write(' normalize_adj: False\n')
        file.write('optim:\n')
        file.write(' optimizer: adam\n')
        file.write(' base_lr: 0.01\n')
        file.write(' max_epoch: ' +  str(configs['max_epoch']) + '\n')
    
    return name + ".yaml"


"""
    Creates the raw dataset that is saved in datasets/raw
"""
def create_cyto_dataset(cyto, eset, cyto_tissue_dict, active_tissue_gene_dict, patient_list, 
                            patient_dict, gene_to_patient, cyto_adjacency_dict):

    # creates graphname
    graphName = cyto + "_" + eset
    #create patientArray
    patientArray = []
    tissues = []


    if cyto == "all":

        for cyto in cyto_tissue_dict.keys():
            # seems to be causing a bug. Rememove UPPer?
            tissues += cyto_tissue_dict[cyto].upper()
        
    else:
        tissues = cyto_tissue_dict[cyto]


    # count the number of active genes in each tissue.
    gene_count = []
    for tissue in tissues:
        if tissue in cyto_adjacency_dict: # the tissue is actually a cytokine.
            count = 1
        else:
            try:
                count = len(active_tissue_gene_dict[tissue.upper()])
            except(KeyError):
                raise(ValueError("{} is in your graph, but is not found in the gene-to-tissue-map".format(tissue.upper())))    
        gene_count.append(count)
    
    total_genes = sum(gene_count)

    active_gene_list = []

    for tissue in tissues:
        for gene in active_tissue_gene_dict[tissue]:
            pair = [gene, tissue]
            active_gene_list.append(pair)


    for patient in patient_list:
        patient_data = {}

        patient_data["DISEASE"] = str(patient_dict[patient])
        patient_data["NAME"] = patient

        data = []
        # create the information vector that goes into each node. 
        for i, tissue in enumerate(tissues): # TODO modify this to avoid stacking?
            tissue_data = [0]*total_genes # initialize an empty vector

            if tissue in cyto_adjacency_dict: # the tissue is actually a cytokine.
                continue                  # we want everything to just be 0 here.

            tissue_genes = active_tissue_gene_dict[tissue]  # get the list of genes that affect the give tissue.

            offset = 0
            #TODO: ADD PCA here.
            for gene in tissue_genes: # set a part of vector data here to data we read, the rest is left as 0
                if(gene == "N/A"): # N/A means "placeholder". We want a slot, but don't have any data for it.
                    continue
                tissue_data[offset] = gene_to_patient[gene][patient] / 20
                offset +=  1
            
            data.append(tissue_data) # add the vector to the matrix.

        patient_data["data"] = data
        patientArray.append(patient_data)

            
    nodeList = cyto_tissue_dict[cyto]
    graphAdjacency = cyto_adjacency_dict[cyto]

    nodeIntMap = {}
    i = 0

    for node in nodeList:
        nodeIntMap[node] = i
        i += 1

    intAdjacency = []
    # turn the adjacency names into int
    for edge in graphAdjacency:
        newEdge = [nodeIntMap[edge[0]], nodeIntMap[edge[1]]]
        intAdjacency.append(newEdge)

    try:
        os.mkdir(os.path.join("datasets"))
    except OSError:
        pass

    try:
        os.mkdir(os.path.join("datasets", graphName))
    except OSError:
        pass

    try:
        os.mkdir(os.path.join("datasets", graphName ,"raw"))
    except OSError:
        pass

    try:
        os.mkdir(os.path.join("datasets", graphName, "processed"))
    except OSError:
        pass
    
    full_dataset = CytokinesDataSet(root="data/", graphName=graphName, filename="full.csv", 
                                    test=True, patients=patientArray, adjacency=intAdjacency, 
                                    nodeNames = nodeIntMap, divisions = gene_count)
    

    torch.save(full_dataset, (os.path.join("datasets", graphName, "raw",  graphName + ".pt")))
    torch.save(active_gene_list, (os.path.join("datasets", graphName, "raw", "geneList.pt")))


def normalize_vector(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    normalized_vector = (vector - min_val) / (max_val - min_val)
    return normalized_vector    

def process_tissues(genes_per_tissue):
    tissue_gene_dict = dict() # maps tissues to the genes associated with them

    gene_set = set()

    tissue_file = open(MAP_FILE)

    tissue_lines = tissue_file.read().splitlines()

    for i in range(0, len(tissue_lines), 2):
        tissue_line = tissue_lines[i]
        
        tissue_line_arr = tissue_line.split(",")

        tissue = tissue_line_arr[0]
        if(len(tissue_lines[i + 1]) > 0):
            genes_array = tissue_lines[i + 1].split(',')

            if genes_per_tissue == 0 or genes_per_tissue > len(genes_array):
                pass
            else:
                random.seed(1234)
                random_indicies = random.sample(range(len(genes_array)), genes_per_tissue)
                genes_array = [genes_array[i] for i in random_indicies]

            tissue_gene_dict[tissue] = genes_array
        else:
            tissue_gene_dict[tissue] = []

        gene_set.update(genes_array)

    return tissue_gene_dict, gene_set

"""
processes eset to retrieve data from patients
"""
def process_eset(eset, gene_set, patient_dict, tissue_gene_dict, cyto_adjacency_dict):
    if(AnnData):
        eset_lines = adp.esetLines()
    else:
        eset_file = open(eset, 'r')
        eset_lines = eset_file.read().splitlines()

    # read the first line, and see if it matches with the patient file provided
    patients = eset_lines[0].replace("\"", "").split(",")[2:]
    
    patient_set = set(patient_dict.keys())

    for patient in patients:
        try:
            patient_set.remove(patient)
        except(KeyError):
            raise(ValueError("{} is not found in the patients file.".format(patient)))


    if (len(patient_set) != 0):
        raise(ValueError("The eset file does not contain {}".format(patient_set)))


    gene_to_patient = dict() # maps the name of a gene to a dict of patients
    for line_num in range(1, len(eset_lines)):
        line = eset_lines[line_num].replace("\"", "")
        parts = line.split(",")
        new_gene = parts[1]


        if (new_gene not in gene_set):
            continue
        # get all the gene expression numbers, and then normalize them
        gene_nums = parts[2:]
        gene_nums = [float(gene_num) for gene_num in gene_nums]
        gene_nums = normalize_vector(gene_nums)


        
        patient_gene_data_dict = dict() # maps the patients code to their gene expression data of this one specific gene
        for index, patient in enumerate(patients):
            patient_gene_data_dict[patient] = gene_nums[index]

        gene_to_patient[new_gene] = patient_gene_data_dict
        
    # make a new tissue_gene_dict of active tisues. If a tissue is in the tissues array, but not in data, do not include it.

    active_tissue_gene_dict = dict()

    for tissue in tissue_gene_dict.keys():
            gene_array = []
            original_genes = tissue_gene_dict[tissue]

            for gene in original_genes:
                if (gene in gene_to_patient.keys() or gene == "N/A"):
                    gene_array.append(gene)
            
            # add PCA
            

            
            active_tissue_gene_dict[tissue] = gene_array
    
    return (gene_to_patient, active_tissue_gene_dict)

def process_graphs(cyto):
    graph_folder_path = "Graphs"
    
    cyto_adjacency_dict = dict() # maps a cytokine's name to their adjacency matrix
    cyto_tissue_dict = dict() # maps a cytokine's name to the tissues they need

    filename = cyto.upper() + "_graph.csv"
    graph_file_path = os.path.join(graph_folder_path, filename)

    graphAdjacency = []
    tissue_set = set()

    f = open(graph_file_path, 'r')
    graphLines = f.read().splitlines()
    
    for line in graphLines:
        parts = line.upper().split(",") # remove newline, capitalize, and remove spaces
        tissue_set.update(parts)
        graphAdjacency.append(parts)
        if sys.argv[3] != "all":
            newParts = [parts[1], parts[0]]
            graphAdjacency.append(newParts)
    

    # put the tissues into a list, and then sort them
    tissue_list = []
    for tissue in tissue_set:
        tissue_list.append(tissue.upper())
    
    tissue_list.sort()

    cyto_adjacency_dict[cyto] = graphAdjacency
    cyto_tissue_dict[cyto] = tissue_list

    return cyto_adjacency_dict,cyto_tissue_dict

"""
Processes Patients to get the list and dictionary of patients
"""
def process_patients(patients):
        if(AnnData):
            patient_lines = adp.patientLines()
        else:
            patient_file = open(patients, 'r')
            patient_lines = patient_file.read().splitlines()

        patient_dict = dict()
        patient_list = []

        for line in patient_lines:
            parts = line.split(",")
            patient_dict[parts[0]] = int(parts[1])
            patient_list.append(parts[0])
        
        return patient_dict, patient_list

def write_lines_to_file(self, input_file, output_file_name):
    try:
        with open(input_file, 'r') as infile:
            # Read all lines from the input file
            lines = infile.readlines()

        # Remove any leading/trailing whitespaces from the output file name
        output_file_name = output_file_name.strip()

        # Write the lines to the output file with the provided name (overwriting if it exists)
        with open(output_file_name, 'w') as outfile:
            for line in lines:
                outfile.write(line)

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")


def make_grid(name, configs, isGraph = True):
    if (not os.path.exists(os.path.abspath("customGrids"))):
        os.makedirs(os.path.abspath("customGrids"))

    # using with statement
    with open(os.path.join('customGrids', name + ".txt"), 'w') as file:
        file.write("dataset.format format ['PyG']\n")
        file.write("dataset.name dataset ['Custom," + name + ",','Custom," + name + ",']\n")
        if (isGraph == True):
            file.write("dataset.task task ['graph']\n")
        else:
            file.write("dataset.task task ['node']\n")
        file.write("dataset.transductive trans [False]\n")
        file.write("dataset.augment_feature feature [[]]\n")
        file.write("dataset.augment_label label ['']\n")
        file.write("gnn.layers_pre_mp l_pre " + str(configs['l_pre']) + "\n")
        file.write("gnn.layers_mp l_mp " + str(configs['l_mp']) + "\n")
        file.write("gnn.layers_post_mp l_post " + str(configs['l_mp']) + "\n")
        file.write("gnn.stage_type stage " + str(configs['stage']) + "\n")
        file.write("gnn.agg agg " + str(configs['agg']) + "\n")



def make_grid_sh(eset_name, cyto, name, save_model):
    grid_sh_path = os.path.join("customScripts", "run_custom_batch_" + eset_name + "_" + cyto + ".sh")
    with open(grid_sh_path, 'w') as file:
        file.write("#!/usr/bin/env bash\n")
        file.write("\n")
        file.write("CONFIG=" + name + "\n")
        file.write("GRID=" + name + "\n")
        file.write("REPEAT=1\n")
        file.write("MAX_JOBS=20\n")
        if(save_model):
            file.write('SAVE=1')
        else:
            file.write('SAVE=0')
        file.write("\n")
        file.write("# generate configs (after controlling computational budget)\n")
        file.write("# please remove --config_budget, if don't control computational budget\n")
        file.write("python configs_gen.py --config configs/${CONFIG}.yaml \\\n")
        file.write(" --config_budget configs/${CONFIG}.yaml \\\n")
        file.write(" --grid customGrids/${GRID}.txt \\\n")
        file.write(" --out_dir configs\n")
        file.write("#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs\n")
        file.write("# run batch of configs\n")
        file.write("# Args: config_dir, num of repeats, max jobs running\n")
        file.write("bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SAVE\n")
        file.write("# rerun missed / stopped experiments\n")
        file.write("bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SAVE\n")
        file.write("# rerun missed / stopped experiments\n")
        file.write("bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SAVE\n")
        file.write("\n")
        file.write("# aggregate results for the batch\n")
        file.write("python agg_batch.py --dir results/${CONFIG}_grid_${GRID}\n")


def make_single_sh(eset_name, cyto, config_name, save_model = False):
    # using with statement
    path = os.path.join("customScripts", "run_custom_" + eset_name + "_" + cyto + ".sh")
    with open(path, 'w') as file:
        file.write('#!/usr/bin/env bash\n')
        file.write('\n')
        escaped_path = os.path.join("configs",config_name).replace("\\", "/")
        file.write('python main.py --cfg ' + escaped_path + ' --repeat 1')

        if(save_model):
            file.write(' --save 1')

        file.write(' --get_edge_weights ')

        if(get_edge_weights):
            file.write('1')
        else:
            file.write('0')


def make_replication(eset_name, cyto, config_name):
    # using with statement
    path = os.path.join("replications", "run_replication_" + eset_name + "_" + cyto + ".sh")
    with open(path, 'w') as file:
        file.write('#!/usr/bin/env bash\n')
        file.write('\n')
        file.write("MODEL_PATH=$1\n")
        escaped_path = os.path.join("configs",config_name).replace("\\", "/")
        file.write('python replication.py --cfg ' + escaped_path + ' --repeat 1 --model_path $MODEL_PATH')


#MAIN
# things we need to be provided
if __name__ == '__main__':

    eset = sys.argv[1]
    eset = os.path.join("rawData", eset + ".csv")

    patients = sys.argv[2]
    patients = os.path.join("rawData", patients + ".csv")

    cyto = sys.argv[3]

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

    # do we save the model
    try:
        save_model = sys.argv[7][0].upper()=="T"
    except(IndexError):
        save_model = False


    try:
        num_genes = int(sys.argv[8][0])
    except(IndexError, ValueError):
        num_genes = 0


    try:
        get_edge_weights = sys.argv[9][0].upper()=="T"
    except:
        get_edge_weights = False


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


    #get patient data
    patient_dict, patient_list = process_patients(patients) # a dict that matches a patient name to their classification

    # process graph data
    cyto_adjacency_dict,cyto_tissue_dict  = process_graphs(cyto) # list of cytokines, maps a cytokine's name to their adjacency matrix, maps a cytokine's name to the tissues they need

    tissue_gene_dict, gene_set = process_tissues(num_genes) # dict that matches tissues to the genes associated with them, a set of all genes we have


    #process eset data
    gene_to_patient, active_tissue_gene_dict = process_eset(eset, gene_set, patient_dict, tissue_gene_dict, cyto_adjacency_dict) # 2 layer deep dict. First layer maps gene name to a dict. Second layer matches patient code to gene expresion data of the given gene.



    # below is for writing new data to stuff, to test multiomnics.
    """file = open("data.csv", mode='w')
    genes = [gene for gene in gene_to_patient.keys()]
    genes.sort()
    tissues = [key for key in active_tissue_gene_dict.keys()]

    for tissue in tissues:
        renamed_tissue = tissue.replace("/", "-")
        file = open("newData/" + renamed_tissue + "_data.csv", mode = 'w')
        genes = [gene for gene in active_tissue_gene_dict[tissue]]
        for patient in patient_dict.keys():
            line = ""
            line += patient + "," + str(patient_dict[patient])
            for gene in genes:
                line += "," + str(gene_to_patient[gene][patient])
            
            line += "\n"
            file.write(line)

    1/0"""

    # below is for writing new data to stuff, to test linear and KNN
    """file = open("data_tcga.csv", mode='w')
    genes = [gene for gene in gene_to_patient.keys()]
    genes.sort()
    tissues = [key for key in active_tissue_gene_dict.keys()]

    for patient in patient_dict.keys():
            line = ""
            line += patient + "," + str(patient_dict[patient])
            for gene in genes:
                line += "," + str(gene_to_patient[gene][patient])
            
            line += "\n"
            file.write(line)


    1/0"""
    # Now, I need to make a new list of every tissue


    # look at gene_to_patient and patient_dict
    # turns the information above into the dataset in the dataset/raw directory.
    create_cyto_dataset(cyto, eset_name, cyto_tissue_dict, active_tissue_gene_dict, patient_list, 
                                patient_dict, gene_to_patient, cyto_adjacency_dict)

    name = cyto + "_" + eset_name

    # below is good. Keep it.

    # All the reading and writing to the dataset needs to be redone. Wonderful.

    # In fact, I wager I probably need a new dataset

    if (grid) :
        make_grid_sh(sys.argv[1], cyto, name, save_model)
        make_grid(name, configs)
        makeConfigFile(name, single_configs)
    else:
        config_name = makeConfigFile(name, configs)
        make_single_sh(sys.argv[1], cyto, config_name, save_model)

        # write a replication script if we save the model
        if(save_model):
            make_replication(sys.argv[1], cyto, config_name)
    #also need to make the grid file and the sh file