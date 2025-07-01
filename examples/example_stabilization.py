import os
from neuraldecoding.dataset import Dataset
from neuraldecoding.stabilization.latent_space_alignment import latent_space_alignment as lse
from neuraldecoding.stabilization.latent_space_alignment import dim_red
from neuraldecoding.stabilization.latent_space_alignment import alignment

## To stabilize
subject = 'Joker'
date_0 = '2024-08-23'
date_k = '2024-11-11'
runs = [1, 2]

ds_0 = Dataset()
ds_0.load_data(subject_name=subject, date=date_0, runs=runs)

ds_k = Dataset()
ds_k.load_data(subject_name=subject, date=date_k, runs=runs)

stabilization = lse.LatentSpaceAlignment(dim_red_method = dim_red.FactorAnalysis(), 
                                         alignment_method = alignment.ProcrustesAlignment(), 
                                         ndims = 10)

ls_0 = stabilization.fit(ds_0)
# model.train(ls_0)

ls_k = stabilization.extract_latent_space(ds_k)
# y, yhat = model.run(ls_k)


## 
