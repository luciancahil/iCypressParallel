import anndata as ad

luad = ad.read_h5ad("./rawData/bulk_mrna_luad.h5ad")

y_data_frame = luad.obs.y

gene_data_frame = luad.var

num_patients = len(y_data_frame)

"""
print(num_patients)
print(y_data_frame.iloc[0])
print(y_data_frame.index[0])

"""

# access gene expression data
#print(luad[0, :].X)

print(luad.var.iloc[0])
