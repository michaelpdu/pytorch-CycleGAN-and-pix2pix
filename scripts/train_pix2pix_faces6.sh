set -ex
python train.py --dataroot ./datasets/faces3 --name face_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --gpu_ids="1" --output_nc=3 --load_size=256 --crop_size=256
