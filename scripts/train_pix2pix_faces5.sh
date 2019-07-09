set -ex
python train.py --dataroot ./datasets/faces2 --name figure_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --gpu_ids="0" --output_nc=3 --load_size=512 --crop_size=512
