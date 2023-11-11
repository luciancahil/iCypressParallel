# iCypressParallel

## What is iCYPRESS?
iCYPRESS stands for identifying CYtokine PREdictors of diSeaSE.

It is a graph neural network library that analyzes gene expression data in the context of cytokine cellular networks.

## Installation and Setup
To Install iCYPRESS, git clone this repository. Then, enter this project's main directory, and run the following commands to create the conda environment needed to make the project run.


````
sed -i 's/\r$//' environment.yml
conda env create -f environment.yml -n cypress-env
conda activate cypress-env
sed -i 's/\r$//' install.sh
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

### Cytokines

There are 70 different cytokines that this network can use to build the Graph in its Graph Neural Network.

The example above uses CCL1, but you can also use CCL2, CD70, or many others.

## Parallelization

Parallization is handled automatically by this library. The last command above dictates wether a single run will be done, or wether a grid search over many possible hyperparameters will be done. To run the library with parallel data using the supplied files, use the following command.

````
cd run
bash custom.sh ESET PATIENTS CCL2 True
````


### Repo Setup on HPC (UBC ARC Sockeye)
To run this library on UBC Sockeye, enter a scratch folder, clone this library, enter the main directory, and then follow the same steps as in installation