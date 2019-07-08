set -ex
python train.py --dataroot ./datasets/faces4 --name faces4_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --gpu_ids="3" --output_nc=1
