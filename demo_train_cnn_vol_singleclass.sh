mkdir ./models
th scripts/train_PTN.lua --gpu 4 --niter 100 --lambda_vox 1 --lambda_msk 0

