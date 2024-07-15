import anndata as ad
import os
import numpy as np

class AnnDataProcessor:
    def __init__(self, filename):
        luad = ad.read_h5ad(filename)
        self.luad = luad
        try:
            self.y_data_frame = luad.obs.y
        except:
            self.y_data_frame = luad.obs.response
        self.num_patients = len(self.y_data_frame)

        # get list of genes.
        self.geneArray = luad.var.to_numpy()

        self.geneArray = [x[0] for x in self.geneArray]

        print("Initialized Anndata")

    """
    Returns a list of patient lines structured as if they were just read from a csv
    """
    def patientLines(self):
        classifaction_map = {} # maps the string describing the ailment to the number its assigned to.
        lines = []
        for i in range(self.num_patients):
            classification = self.y_data_frame.iloc[i]

            if classification not in classifaction_map: #check to see if we have seen this diagnosis
                # if we haven't seen this classifcation, add it to the map.
                class_number = len(classifaction_map)

                classifaction_map[classification] = class_number
            
            patient = self.y_data_frame.index[i]
            number = classifaction_map[classification]

            line = patient + "," + str(number)
            lines.append(line)

        return lines
    
    """Returns a list of eset lines as if they were just read from a csv"""
    def esetLines(self):
        lines = []

        # write the first line:
        line = "\"\",\"gene_sym\""

        for i in range(self.num_patients):
            patient = self.y_data_frame.index[i]

            line += ",\"" + patient + "\""
        
        lines.append(line)

        #write the lines for each gene symbol

        # we need to create a matrix of data, then transpose it.
        data_matrix = []
        for i in range(self.num_patients):
            data_matrix.append(np.array(self.luad[i, :].X[0]))


        data_matrix = [list(row) for row in zip(*data_matrix)] # transpose it.

        for i, data_line in enumerate(data_matrix):
            line = "\"" + str(i + 1) +  "\",\"" + self.geneArray[i] + "\"," + str(data_line)[1:-1]
            lines.append(line)
        
        return lines
        





"""
print(num_patients)
print(y_data_frame.iloc[0])
print(y_data_frame.index[0])

"""

# access gene expression data
#print(luad[0, :].X)

"""path = os.path.join("rawData", "bulk_mrna_luad.h5ad")

adp = AnnDataProcessor(path)

adp.esetLines()"""