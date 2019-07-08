set -ex
python train.py --dataroot ./datasets/maps --name maps_pix2pix_2 --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --gpu_ids="3" --preprocess=none
