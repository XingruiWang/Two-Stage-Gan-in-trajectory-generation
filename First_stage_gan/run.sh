CUDA_VISIBLE_DEVICES=0,1,2,3 python WGANGP.py \
--dataroot ./grid32/ \
--labelroot ./traj_all_0115.txt \
--outf ./output \
--batchSize 64 \
--n_critic 1 \
--netG ./output_IN/netG_epoch_320.pth \
--netD ./output_IN/netD_epoch_320.pth \
--cuda \
--start_iter 320 \
--niter 350
