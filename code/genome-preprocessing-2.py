import numpy as np
import sgkit as sg
from sgkit.io.vcf import vcf_to_zarr
import os

data_dir = "../data/datasets/1kgp_201408_genotype"
data_file = "ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped.vcf.gz"

# convert to zarr
sg.io.vcf.vcf_to_zarr(os.path.join(data_dir, data_file), "output.zarr")
ds = sg.load_dataset("output.zarr")

# create an array we can use
# replace missing values with zero
# (deeper analysis may require some form of imputation)
gt_values_replaced = ds.call_genotype.values
gt_values_replaced[gt_values_replaced==-1] = 0

del(ds)

gt_sum = np.sum(gt_values_replaced, axis=2)

del(gt_values_replaced)

gt_sum = gt_sum.T

# output to compressed numpy file
np.savetxt("gt_sum_thinned.npy.gz", gt_sum)