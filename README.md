# iCypressParallel

## What is iCYPRESS?
iCYPRESS stands for identifying CYtokine PREdictors of diSeaSE.

It is a graph neural network library that analyzes gene expression data in the context of cytokine cellular networks.

## Installation and Setup
To Install iCYPRESS, git clone this repository. Then, enter this project's main directory, and run the following commands to create the conda environment needed to make the project run.

Alternatively, run the following commands in the folder where you want the project to live:

````
git clone https://github.com/luciancahil/iCypressParallel.git
cd iCypressParallel
conda create -n cypress python=3.7 -y
conda activate cypress
pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
CUDA=cu101
TORCH=1.8.0
pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==2.2.0
pip install anndata==0.8.0
pip install matplotlib==3.5.3
pip install jupyter==1.0.0
bash install.sh
````

*If you see "ERROR: No matching distribution found for matplotlib==3.7.0", just ignore it. The library will still work fine*

## Testing

To make sure that all packages are installed properly type the following commands into console, starting at the main directory.

````
cd run
bash custom.sh GSE40240_eset GSE40240_patients CCL2 False
````


## Arguments explained

There are seven arguments that can be passed to custom.sh in the command line. In order, they are:

````
bash custom.sh ESET PATIENT GRAPH GRID MAP HYPERPARAM SAVE
````

Of these arguments, the first 4 are mandatory, while the other 3 are optional.


The arguments in order:

|Parameter       |  Explanation   |  
| ---------------|----------------|
| ESET           | This has 2 modes. If the word "ann" is passed, this library will read input in the form of AnnData (see "Using anndata" below for more details). Otherwise, if the name of an ESET file is passed, this will the file where all input data will be read from (See "Custom Data" below for more details.)|
| PATIENT        |This also has 2 modes. If the word "ann" was passed to eset, then pass the name of the annfile to this parameter (see "Using anndata" below for detalis). Otherwise, pass the name of a file containing patient partition data (see "Custom Data" below for details).|
| GRAPH           | Pass the name of the cytokine who's biological information will be used to build the graph structure. See "Graphs" below for details.    |
| GRID           | Pass either "True" or "False" to this parameter. If true is passed, then the library will perform a grid search, trying multiple different hyper parameters. If "False" is passed, then only one set of hyperparameters will be run. See "Paralllelization" below for more details.    |
| MAP            | Optional. Pass the name of the file that controls how eset data is mapped to nodes to this. See "Custom Mapping" below for details.    |
| HYPERPARAM     | Optional. The file that controls the hyperpameters of this the neural network created. See "Custom Mapping" below for that.    |
| SAVE           | Optional. Pass either "True" or "False" to this parameter. If "True" is passed, the program will save the model after is done training. See "Saving" below for details    |
| NUM_GENES      | Optional. The number of genes we randomly select to use, in order to prevent memory errors. Pass either a positive integer or 0. If 0 is passed, all genes will be used. If a positive number is passed, any tissue with more than that number of genes will have a random sample of genes included.

If you wish to pass a parameter to either SAVE or HYPERPARAM but wish to just use the default value for all optional parameters before it, pass "NULL" into the parameters as a placeholder. For example, to save the model, but use default map and hyperparameters, use the following command:

````
bash custom.sh GSE40240_eset GSE40240_patients CCL2 False NULL NULL True
````


## Custom Data

To use this library on your own custom data, you will need two csv files: a patients file, and an eset file.

The eset file should be structured like so:

|"             "|  "gene_sym" |"GSM989153"  |"GSM989154" |"GSM989155" |
| ------------- |-------------| -----       | ---        | ---        |
| "1"           | "A1CF"      | 3.967147246 |3.967147248 |3.96714725  |
| "2"           | "A2M"       | 4.669213864 |4.669213567 |4.669213628 |
| "3"           | "A2ML1"     | 4.140074251 |4.140074246 |4.140074286 |


The top should include every patient who's data you wish to analyze, the 2nd collumn should contain 5h3 name of every gene you have data on, and the numbers represent the gene readings.

Meanwhile, the patients file should be structured like so:
|         |   |
|---------|---|
|GSM989161|0  |
|GSM989162|0  |
|GSM989163|0  |
|GSM989164|0  |
|GSM989165|0  |
|GSM989166|0  |
|GSM989167|1  |
|GSM989168|1  |
|GSM989169|1  |
|GSM989170|1  |
|GSM989171|1  |
|GSM989172|1  |
|GSM989173|1  |
|GSM989174|1  |
|GSM989175|1  |

Where the left collumn contains all the names of your patients, and the right collumn contains their classification.

It is very important that every patient that appears in your patients file also appears in the top row of your eset file and vice versa. The files also must stored in csv format. Otherwise, the library will raise an error.

Once you've prepared both files, place them into the run/rawData directory.

You can then analyze the files using this library with one single command. Run the command above, but replace the first and second arguments with the name of your files, minus the suffix. 

For example, if your eset file is called "ESET.csv", and your patient file is called "PATIENTS.csv", run the following command. 

````
cd run
bash custom.sh ESET PATIENTS CCL2 False
````

## Using anndata

To use Anndata files (.h5ad files) instead of .csv files, ensure the following rules are followed:

- All Data Should be in a single .h5ad file.
- The .h5ad file should contain a single AnnData object of dimensions n_obs × n_vars, where n_obs is the number of patients, and n_vars is the number of genes you have data on.
- The list of gene names should be in the .var element.
- The classification of each patient should be in .y variable, accessible by y.iloc
- The data for each patient in each gene should be stored in the .X variable

To use anndata, place the .h5ad file in the rawData folder, and use the following command. (Replace ann_data_file_name with the actual name of the file)

````
bash custom.sh ANN ann_data_file_name CCL2 False
````
For example, if you want to use CCL2 and you had an anndata file called info.h5ad, use the following command:

````
bash custom.sh ANN info CCL2 False
````

### Graphs

There are 70 different cytokines that this network can use to build the Graph in its Graph Neural Network.

The example above uses CCL2, but you can also use CCL1, CD70, or many others. Replace CCL2 with the name of the cytokine you want to observe.

If you instead want to use a giant network containing all cytokines, use the following command:

````
bash custom.sh ESET PATIENTS all False
````

If you had an anndata file called "data.h5ad", use the following command:

````
bash custom.sh ANN info all False
````

### Custom Graphs

To use a custom graph, add a csv file with the suffix "_graph.csv" to the run/graphs folder. For example. If you want to add a graph called "DUMMY", add a csv file into the graphs folder called "DUMMY_graphs.csv". The format is important. It must end with _graphs.csv, and all letters before that must be capitalized. Otherwise, it will not show up.

The csv should be structured with 2 collumns. The first collumn will be the source of a connection, and the second the destination of a particular connection. For example, see CCL1_graph.csv:

`````
intermediate monocyte,T-reg
classical monocyte,T-reg
`````

This graph has 3 nodes and 2 edges; one edge starts at intermediate monocyte and the other starts at classical monocyte, while both end at T-reg.

Make sure that any node name you have in your graph file shows up in your mapping file (see "Custom Mapping" below for more details).

## Parallelization

Parallization is handled automatically by this library. The last command above dictates wether a single run will be done, or wether a grid search over many possible hyperparameters will be done. To run the library with parallel data using the supplied files, use the following command.

````
cd run
bash custom.sh ESET PATIENTS CCL2 True
````

## Custom Mapping

The mapping command specifies what kind of data goes into a node. By default, the library uses the file "GenesToTissues.csv".

Here is a sample:

````
PLASMACYTOID DC
ALOX5AP,APP,BCL11A,BDCA2,BDCA4,C10ORF118,C12ORF75,...
T-REG
AB22510,ABCAM,AIRE,ARID5B,BACH2,BATF,CAPG,CCR10,...
CCL26
````

Here, Plasmacytoid DC and T-Reg are potential names for nodes that a graph might have, and ALOX5AP and CAPG are names for the types of data that may go into them.

If you want to make your own file, follow the structure of alternating Node names in one line and data point name array in another. Each element in the array must be seperated by a comma.

If you want a node to have one space dedicated to it in the vectors the graph creates, have a data point name "N/A". This way, one space will be allocated just for that node, but it won't have any data.

Any node name must appear in this file. If you deleted the 2 lines corresponding to T-REG, and then tried to use a cytokine that involves T-REG, the progam will throw an error. 

It's fine if a data name doesn't appear in your list, though. For instance, if a data set has the name of a gene that isn't in "GeneToTissues.csv", the library will run, the gene will simply be ignored.

To build a dataset using your custom mapping, simply use the following command, which adds a parameter onto the command:

````
bash custom.sh ESET PATIENTS all False MAP_FILE
````

Include the suffix. For example, the following will just use the default mapping file:

````
bash custom.sh GSE40240_eset GSE40240_patients all False GenesToTissues.csv
````

## Custom Hyperparameters

Hyperparameters specify the structures of the Neural network. Things like how many epochs to train for, how many neurons in a hidden layer, etc. this library allows one to play around with the hyperparameters of the network by making simple changes to yaml files.

The default hyperparameters are in the "run/Hyperparameters" directory. Any custom hyperparameter configs should be added to that folder.

### Single Run

For a single run, the hyperparameters stickly handle the structure of the Graph Neural network. The default values, found in "run/Hyperparameters/Default Configs.yaml" are as follows:

````
batch_size : 80         # When training, how many instances should we have at once
eval_period : 20        # After how many training rounds we do an evaluation
layers_pre_mp : 2       # How many layers exist before the message passing phase
layers_mp : 6           # How many rounds of message passing do we do
layers_post_mp : 2      # HOw many layers deep is the neural network after we do all the message passing
dim_inner : 137         # How many neurons are in the hidden layers
max_epoch : 400         # How many epochs we train for
````

If you wish to change these hyperparameters, it is recommended that you make a new file. The new file must be of .yaml type. For instance, if you wanted to train for 500 epochs, you could make a new file called "Train_500.yaml" with the following content:

````
batch_size : 80         # When training, how many instances should we have at once
eval_period : 20        # After how many training rounds we do an evaluation
layers_pre_mp : 2       # How many layers exist before the message passing phase
layers_mp : 6           # How many rounds of message passing do we do
layers_post_mp : 2      # HOw many layers deep is the neural network after we do all the message passing
dim_inner : 137         # How many neurons are in the hidden layers
max_epoch : 500         # How many epochs we train for
````

To run the network with the new hyperparameter, place that file in the "run/Hyperparameters" directory, and run the following command from the run directory:

````
bash custom.sh ESET PATIENTS CCL2 False MAP_FILE Train_500
````
If you wish to just use the default mapping file, write "null" (case-insensitive) into the MAP_FILE argument. 



Note: If you don't see the new hyperparameters taking affect, it may because there is already a config file with the same name as the one the library is trying to generate. Clearing the configs folder should resolve this issue.

### Parallel Runs

Parallel runs work a little differently. Since they by design try mupltiple different configurations for the network, hyperparameters here specify the search space.

The default search space is specified in "run/Hyperparameters/Default Grid.yaml":

````
l_pre : [1,2]                           # How many layers exist before the message passing phase
l_mp : [2,4,6,8]                        # How many rounds of message passing do we do
l_post : [2,3]                          # How many layers deep is the neural network after we do all the message passing
stage : ['skipsum','skipconcat']        # How Staging is achieved
agg : ['add','mean']                    # How data in each hidden layer is pooled after message passing is finished

````

If you wish to change these hyperparameters, it is recommended that you make a new file. The new file must be of .yaml type. For instance, if you wanted to only try adding as the agg function, you could make a new file called "add_only.yaml" with the following content:

````
l_pre : [1,2]                           # How many layers exist before the message passing phase
l_mp : [2,4,6,8]                        # How many rounds of message passing do we do
l_post : [2,3]                          # How many layers deep is the neural network after we do all the message passing
stage : ['skipsum','skipconcat']        # How Staging is achieved
agg : ['add']                           # How data in each hidden layer is pooled after message passing is finished
````
To run the network with the new hyperparameter, place that file in the "run/Hyperparameters" directory, and run the following command from the run directory:

````
bash custom.sh ESET PATIENTS CCL2 True add_only
````

Note: If you don't see the new hyperparameters taking affect, it may because there is already a config file with the same name as the one the library is trying to generate. Clearing the grid folder should resolve this issue.

## Saving

To save, simply pass "True" as the sixth parameter. If you wish to use custom hyperparameters, simply add "True" to the end of the command.

If you wish to use default hyperparameters and mapping, insert "NULL" in place of the hyperparameter command.

````
bash custom.sh GSE40240_eset GSE40240_patients CCL1 False NULL NULL True
````

## Replication

To use a model you've saved, do the following.

1. Note the path the model is saved to by looking at the terminal. You should see a message "Model saved at [PATH]".
1. Write a new command with the same 4 parameters as your custom.sh command, but with "custom.sh" replaced with "replication.sh".
1. Add the path you copied as a 5th parameter to the command. Paste that command in terminal, and you should see the output of the network on each data point.

If you wanted to try the code on new input data, simply replace the first 2 parameters with what you wanted. However, the structure must be the same; same number of elements in each vector, same graph structure, etc.

## Analysis

The library also comes packaged with ways to evaluate the models that a grid search generates, and visualize them via violin plots.

To start, install the following library in your conda env

````
pip install seaborn==0.12.2
````



To use this functionality, edit the file "analysis/example.pynb" by creating a newcell with the following format:

````
experiment_name = '[NAME]'
dataset = '[DATASET]'
plot_analysis(experiment_name, division='val', dataset=dataset)
````

To find the correct value of the experiment_name parameter, go to "run/results" and see the name of the newly generated folder. Copy the name of the new folder into the "experiment_name" field above. For "dataset", do the same, but go to the "run/datasets" folder.

For example, after a run, I had a new folder named "CCL2_GSE40240_eset" in the datasets file, and a new folder named "CCL2_GSE40240_eset_grid_CCL2_GSE40240_eset" in the results folder. Thus, I wrote the following cell:

````
experiment_name = 'CCL2_GSE40240_eset_grid_CCL2_GSE40240_eset'
dataset = 'CCL2_GSE40240_eset'
plot_analysis(experiment_name, division='val', dataset=dataset)
````

Explanation of the parameters: 
|         |   |
|---------|---|
| Name | Explanation|
| experiment_name| Name of the experiment that was just run. Autogenerated for each experiment.|
| dataset | Name of the dataset that the experiment ran on. Autogenerated by the library. |
| division | Either "train" or "test". How the various structures will be ranked. Recommended to be set to "val".|

Change the 

### Example

Let's say you started by running the following command:

````
bash custom.sh GSE40240_eset GSE40240_patients CCL2 False NULL NULL True
````

Then, you got the following message at the end of the runtime:

````
Model saved at models/CCL2_GSE40240_eset_1711341524.4664948_model.pt
````

You would then run the following command to try a replication:

````
bash replication.sh GSE40240_eset GSE40240_patients CCL2 False models/CCL2_GSE40240_eset_1711341524.4664948_model.pt
````

If you wanted to try the code on new input data, simply replace the first 2 parameters with what you wanted. However, the structure must be the same; same number of elements in each vector, same graph structure, etc.


## Results


### Single

### Parallel

The last text output will be something like this:

```
Results aggregated across models saved in results/CCL2_ANN_grid_CCL2_ANN/agg
```

To look at the results of the entire go into the new results folder, then enter the "agg" folder. 


You will see 6 files.

Train and Val will show how the networks performed in their last training and testing epochs respectively network.

Train_best and Val_best will show the best single run of each network.

Train_best_epoch and Val_best_epoch will show the best epoch of each network.

To see  more detailed results, go back to the parent folder of agg ("CCL2_ANN_grid_CCL2_ANN" in the above example).

You can now enter any subfolder for more information. 

At the first level, you can see the config.yaml file for that specific configuartion.

Enter the 0 folder, and you will see the train and val folders for that network.

Enter either, and you will see "stats.json". The folder will contain all the information about every training and testing run done by this network as it was being optimized.


## Repo Setup on HPC (UBC ARC Sockeye)
To run this library on UBC Sockeye, enter a scratch folder, clone this library, enter the main directory, and then follow the same steps as in installation


## Node Classification

To run a node classification job, use the following:


````
bash custom_node.sh [NODE_FILE] [EDGE_FILE] [NAME] [GRID]
````
The arguments in order, with the first 4 being mandatory:

|Parameter       |  Explanation   |  
| ---------------|----------------|
| NODE_FILE      | The name of the file that contains information about the nodes.|
| EDGE_FILE      | The name of the file that contains infomration about the edges.|
| NAME           | Anthing goes here. Just the name of the project. Again, any name works  |
| GRID           | Pass either "True" or "False" to this parameter. If true is passed, then the library will perform a grid search, trying multiple different hyper parameters. If "False" is passed, then only one set of hyperparameters will be run. See "Paralllelization" below for more details.    |
| HYPERPARAM     | Optional. The file that controls the hyperpameters of this the neural network created. See "Custom Mapping" below for that.    |
| SAVE           | Optional. Pass either "True" or "False" to this parameter. If "True" is passed, the program will save the model after is done training. See "Saving" below for details    |

For example, to run a job with "CORA", try below.
````
bash custom_node.sh Cora-nodes Cora-edges CORA False
````

### Structuring Files for Node Classification

#### Nodes

The nodes should be structured in a CSV, with each row representing one node. In each row, the first element represents the class that node is, and every element after should represent an element in the input vector. See "run/rawData/Cora-nodes.csv" for an example.

For edge purposes, the first row is called node "0", then node "1", and so on.

#### Edges

The edges should be a csv. Each line is one edge. Each line shoudl only have 2 elemnets, with the first element representing the source, and the send element represeting the destination.

See "run/rawData/Cora-edges.csv"

## Jupyter Notebook on Sockeye

To use Jupyter Notebook on sockey, first, pull a custom docker image into your project file. Then, clone the follwing docker image from docker hub:

````
module load gcc
module load apptainer
cd /arc/project/<st-alloc-1>/jupyter
apptainer pull --force --name cypress-docker.sif docker://royhe62/cypress-docker
````

Then, create a folder in the stack folder. Write this script into your scratch project folder, and then start a job using sbatch.

````
#!/bin/bash
 
#SBATCH --job-name=my_jupyter_notebook
#SBATCH --account=<st-alloc-1>
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
 
################################################################################
 
# Change directory into the job dir
 
# Load software environment
module load gcc
module load apptainer
 

export TMPDIR=/scratch/<st-alloc-1>/jupyter
 

# Set RANDFILE location to writeable dir
export RANDFILE=$TMPDIR/.rnd
  
# Generate a unique token (password) for Jupyter Notebooks
export APPTAINERENV_JUPYTER_TOKEN=$(openssl rand -base64 15)
 
# Find a unique port for Jupyter Notebooks to listen on
readonly PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
 
# Print connection details to file
cat > connection_${SLURM_JOB_ID}.txt <<END
 
1. Create an SSH tunnel to Jupyter Notebooks from your local workstation using the following command:
 
ssh -N -L 8888:${HOSTNAME}:${PORT} ${USER}@sockeye.arc.ubc.ca
 
2. Point your web browser to http://localhost:8888
 
3. Login to Jupyter Notebooks using the following token (password):
 
${APPTAINERENV_JUPYTER_TOKEN}
 
When done using Jupyter Notebooks, terminate the job by:
 
1. Quit or Logout of Jupyter Notebooks
2. Issue the following command on the login node (if you did Logout instead of Quit):
 
scancel ${SLURM_JOB_ID}

END

apptainer exec --home /scratch/<st-alloc-1>/my_jupyter --env XDG_CACHE_HOME=/scratch/<st-alloc-1>/my_jupyter /arc/project/<st-alloc-1>/jupyter/cypress-docker.sif jupyter notebook --no-browser --port=${PORT} --ip=0.0.0.0 --notebook-dir=$SLURM_SUBMIT_DIR

````

Once the job starts, a txt file with instructions on how to connect to the Jupyter server will be generated. Follow those instructions to connect.

Once a connection has been made, you may use Jupyter anyway you wish. See run/demonstration.pynb for several examples of what can be done.