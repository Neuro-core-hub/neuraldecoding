import latent_space_alignment
## To stabilize

ds_0 = ...
ds_k = ...
cfg = ...
model = cfg.model

stabilization = latent_space_alignment(dim_red_method = cfg.dim_red_method, alignment_method = cfg.alignment_method, ndims = cfg.ndims)

ls_0 = stabilization.fit(ds_0)
model.train(ls_0)

ls_k = stabilization.extract_latent_space(ds_k)
y, yhat = model.run(ls_k)


## 
