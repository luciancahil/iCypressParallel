import anndata as ad
import os






class AnnDataProcessor:
    def __init__(self, filename):
        luad = ad.read_h5ad(filename)
        self.luad = luad
        self.y_data_frame = luad.obs.y
        self.num_patients = len(self.y_data_frame)

        # get list of genes.
        self.geneArray = luad.var.to_numpy()

        self.geneArray = [x[0] for x in self.geneArray]

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



"""
print(num_patients)
print(y_data_frame.iloc[0])
print(y_data_frame.index[0])

"""

# access gene expression data
#print(luad[0, :].X)

path = os.path.join("rawData", "bulk_mrna_luad.h5ad")

adp = AnnDataProcessor(path)

adp.patientLines()