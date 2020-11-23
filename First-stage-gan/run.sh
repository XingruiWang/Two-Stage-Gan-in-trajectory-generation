#CUDA_VISIBLE_DEVICES=0 python WGANGP_grid_data1-0128.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 python WGANGP_Lnorm.py \
--dataroot ./grid32_0304/ \
--labelroot ./traj_all_0115.txt \
--outf ./output_LN \
--cuda
