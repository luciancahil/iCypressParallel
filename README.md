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
pip install -r requirements.txt
pip install anndata
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


### Repo Setup on HPC (UBC ARC Sockeye)
To run this library on UBC Sockeye, enter a scratch folder, clone this library, enter the main directory, and then follow the same steps as in installation
