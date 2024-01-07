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

        print(self.geneArray)




"""
print(num_patients)
print(y_data_frame.iloc[0])
print(y_data_frame.index[0])

"""

# access gene expression data
#print(luad[0, :].X)

path = os.path.join("rawData", "bulk_mrna_luad.h5ad")

adp = AnnDataProcessor(path)