set -ex
python test.py --dataroot ./datasets/maps --name mapx_pix2pix --model pix2pix --netG unet_256 --direction AtoB --dataset_mode aligned --norm batch
