import os
import shutil
import argparse
import platform
from data import create_dataset
from models import create_model
from PIL import Image
from tools.face_helper import save_face_image, GAP
from tools.image_helper import generate_fake_merged_image, crop_square, ImageHelper

class DepthGenerator:
    def __init__(self):
        self.tmp_dir = 'tmp_dir'
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        self.tmp_sample_dir = os.path.join(self.tmp_dir, 'samples')
        self.tmp_test_sample_dir = os.path.join(self.tmp_dir, 'samples', 'test')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        if not os.path.exists(self.tmp_test_sample_dir):
            os.makedirs(self.tmp_test_sample_dir)
        self.tmp_results_dir = os.path.join(self.tmp_dir, 'results')
        self.gen_face = False

        self.image_helper = ImageHelper()

        platform_system = platform.system() 
        if platform_system == 'Linux':
            self.python_path = "/home/dupei/anaconda3/envs/pix2pix/bin/python"
        elif platform_system == 'Darwin':
            self.python_path = "/Users/dupei/anaconda3/envs/pytorch/bin/python"
        else:
            print('ERROR: Unsupported platform!')

    def __del__(self):
        # if os.path.exists(self.tmp_dir):
        #     shutil.rmtree(self.tmp_dir)
        pass

    def run_test_script(self, sample_dir, model_name, result_dir):
        cmd = '{} test.py --dataroot={} --name={} --model=pix2pix --netG=unet_256 --direction=AtoB --dataset_mode=aligned --norm=batch --output_nc=3 --gpu_ids="-1" --results_dir={} --load_size=512 --crop_size=512'.format(self.python_path, sample_dir, model_name, result_dir)
        os.system(cmd)

    def gen_image_depth(self, input_file):
        # generate fake merged input_file to tmp_dir
        dir_path, name = os.path.split(input_file)
        name_wo_ext, ext = os.path.splitext(name)
        merged_figure_file = os.path.join(self.tmp_test_sample_dir, name_wo_ext+'_fake_merged'+ext)
        generate_fake_merged_image(input_file, merged_figure_file)

        # generate depth image of figure
        self.run_test_script(self.tmp_sample_dir, 'figure_pix2pix', self.tmp_results_dir)
        os.remove(merged_figure_file)

        figure_depth_path = os.path.join(self.tmp_results_dir, 'figure_pix2pix', 'test_latest', 'images',
                                            name_wo_ext + '_fake_merged_fake_B.png')

        if self.gen_face:
            # get face information
            # 1. crop square image from 640X480, so cropped image is 480X480
            square_file = os.path.join(self.tmp_sample_dir, name_wo_ext + '_square' + ext)
            crop_square(input_file, square_file)
            # 2. get face image from 480X480 image，and return face position
            face_file = os.path.join(self.tmp_sample_dir, name_wo_ext+'_face'+ext)
            (left, top, right, bottom) = save_face_image(square_file, face_file)
            # 3. generate fake merged face images
            merged_face_file = os.path.join(self.tmp_test_sample_dir, name_wo_ext + '_face_fake_merged' + ext)
            generate_fake_merged_image(face_file, merged_face_file)
            # 4. remove square file and face file
            os.remove(square_file)
            os.remove(face_file)

            # generate depth image of face
            self.run_test_script(self.tmp_sample_dir, 'face_pix2pix', self.tmp_results_dir)
            os.remove(merged_face_file)

            # merge face into figure image
            face_depth_path = os.path.join(self.tmp_results_dir, 'face_pix2pix', 'test_latest', 'images',
                                            name_wo_ext + '_face_fake_merged_fake_B.png')

        # prepare a new black image 640x480
        image = Image.new('RGB', (640, 480))
        # resize generated figure image to 480x480
        figure_image = Image.open(figure_depth_path)
        figure_image = figure_image.resize((480, 480))
        if self.gen_face:
            # resize generated face image to original size
            face_image = Image.open(face_depth_path)
            face_image = face_image.resize((right-left, bottom-top))
            figure_image.paste(face_image, (left, top, right, bottom))
        # paste figure image to black one
        image.paste(figure_image, (80, 0, 480+80, 480))
        rgb_depth_path = os.path.join(self.tmp_results_dir, name_wo_ext + '_depth.png')
        image.save(rgb_depth_path)
        # convert RGB888 to 8-bit grayscale
        depth_8bit_path = os.path.join(self.tmp_results_dir, name_wo_ext + '_depth_8bit.png')
        self.image_helper.rgb888_to_8bit_grayscale(rgb_depth_path, depth_8bit_path)
        # convert 8-bit grayscale to 16-bit grayscale
        depth_16bit_path = os.path.join(self.tmp_results_dir, name_wo_ext + '_depth_16bit.png')
        self.image_helper.grayscale_8to16(depth_8bit_path, depth_16bit_path)

    def generate(self, input):
        if os.path.isfile(input):
            self.gen_image_depth(input)
        elif os.path.isdir(input):
            pass
        else:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of DepthImageGenerator')
    parser.add_argument("input", type=str, help="input image file")
    args = parser.parse_args()
    gen = DepthGenerator()
    gen.generate(args.input)
