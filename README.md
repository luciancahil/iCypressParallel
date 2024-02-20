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
conda create -n cypress-env python=3.7 -y
conda activate cypress-env
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

### Cytokines

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

## Parallelization

Parallization is handled automatically by this library. The last command above dictates wether a single run will be done, or wether a grid search over many possible hyperparameters will be done. To run the library with parallel data using the supplied files, use the following command.

````
cd run
bash custom.sh ESET PATIENTS CCL2 True
````

## Custom Hyperparameters

Hyperparameters specify the structures of the Neural network. Things like how many epochs to train for, how many neurons in a hidden layer, etc. this library allows one to play around with the hyperparameters of the network by making simple changes to yaml files.

The default hyperparameters are in the "run/Hyperparameters" directory. Any custom hyperparameter configs should be added to that folder.

### Single Run

For a single run, the hyperparameters stickly handle the structure of the Graph Neural network. The default values, found in "run/Hyperparameters/Default Configs.yaml" are as follows:

````
blood_only : True       # Whether we use tissue types that come from blood when building the networks
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
blood_only : True       # Whether we use tissue types that come from blood when building the networks
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
bash custom.sh ESET PATIENTS CCL2 False Train_500
````

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

If you wish to use default hyperparameters, insert "NULL" in place of the hyperparameter command.

````
bash custom.sh GSE40240_eset GSE40240_patients CCL1 False NULL True
````

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
