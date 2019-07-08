import os
import shutil
import argparse
from data import create_dataset
from models import create_model
from PIL import Image
from tools.face_helper import save_face_image
from tools.image_helper import generate_fake_merged_image

class DepthGenerator:
    def __init__(self):
        self.tmp_dir = 'tmp_dir'
        self.tmp_sample_dir = os.path.join(self.tmp_dir, 'samples')
        self.tmp_test_sample_dir = os.path.join(self.tmp_dir, 'samples', 'test')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        if not os.path.exists(self.tmp_test_sample_dir):
            os.makedirs(self.tmp_test_sample_dir)
        self.tmp_results_dir = os.path.join(self.tmp_dir, 'results')
        if not os.path.exists(self.tmp_results_dir):
            os.makedirs(self.tmp_results_dir)
        self.figure_results_dir = os.path.join(self.tmp_results_dir, 'figure_results')
        self.face_results_dir = os.path.join(self.tmp_results_dir, 'face_results')

        self.opt_figure = self.init_options(self.tmp_sample_dir, 'figure_pix2pix', self.figure_results_dir)
        self.opt_face = self.init_options(self.tmp_sample_dir, 'face_pix2pix', self.face_results_dir)

        self.model_figure = self.setup_model(self.opt_figure)
        self.model_face = self.setup_model(self.opt_face)

    def init_options(self, sample_dir, model_name, output_dir):
        opt = argparse.Namespace()
        # basic parameters
        opt.dataroot = sample_dir
        opt.name = model_name
        opt.gpu_ids = [] # gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU
        opt.checkpoints_dir = './checkpoints' # models are saved here
        # model parameters
        opt.model = 'pix2pix' # chooses which model to use. [cycle_gan | pix2pix | test | colorization]
        opt.input_nc = 3 # num of input image channels: 3 for RGB and 1 for grayscale
        opt.output_nc = 3 # num of output image channels: 3 for RGB and 1 for grayscale
        opt.ngf = 64 # num of gen filters in the last conv layer
        opt.ndf = 64 # num of discrim filters in the first conv layer
        opt.netD = 'basic' # specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator
        opt.netG = 'unet_256' # specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        opt.n_layers_D = 3 # only used if netD==n_layers
        opt.norm = 'batch' # instance normalization or batch normalization [instance | batch | none]
        opt.init_type = 'normal' # network initialization [normal | xavier | kaiming | orthogonal]
        opt.init_gain = 0.02 # scaling factor for normal, xavier and orthogonal.
        opt.no_dropout = False # no dropout for the generator
        # dataset parameters
        opt.dataset_mode = 'aligned' # chooses how datasets are loaded. [unaligned | aligned | single | colorization]
        opt.direction = 'AtoB' # AtoB or BtoA
        opt.serial_batches = True # if true, takes images in order to make batches, otherwise takes them randomly
        # opt.num_threads = 4 # num of threads for loading data
        opt.batch_size = 1 # input batch size
        opt.load_size = 256 # scale images to this size
        opt.crop_size = 256 # then crop to this size
        opt.max_dataset_size = float("inf") # Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.
        opt.preprocess = 'resize_and_crop' # scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
        opt.no_flip = True # if specified, do not flip the images for data augmentation
        opt.display_winsize = 256 # display window size for both visdom and HTML
        # additional parameters
        opt.epoch = 'latest' # which epoch to load? set to latest to use latest cached model
        opt.load_iter = 0 # which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]
        opt.verbose = True # if specified, print more debugging information
        opt.suffix = '' # customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}

        # test parameters
        opt.ntest = float("inf") # num of test examples.
        opt.results_dir = output_dir # saves results here.
        opt.aspect_ratio = 1.0 # aspect ratio of result images
        opt.phase = 'test' # train, val, test, etc
        # Dropout and Batchnorm has different behavioir during training and test.
        opt.eval = False # use eval mode during test time.
        opt.num_test = 50 # how many test images to run

        # hard-code some parameters for test
        opt.num_threads = 0   # test code only supports num_threads = 1
        opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        opt.isTrain = False
        return opt

    def setup_model(self, opt):
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        return model

    def __del__(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def gen_depth_image(self, opt, model):
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference

    def gen_image_depth(self, input_file):
        # generate fake merged input_file to tmp_dir
        dir_path, name = os.path.split(input_file)
        name_wo_ext, ext = os.path.splitext(name)
        merged_figure_file = os.path.join(self.tmp_test_sample_dir, name_wo_ext+'_fake_merged'+ext)
        generate_fake_merged_image(input_file, merged_figure_file)

        # generate depth image of figure
        self.gen_depth_image(self.opt_figure, self.model_figure)

        # get face information
        face_file = os.path.join(dir_path, name_wo_ext+'_face'+ext)
        (left, top, right, bottom) = save_face_image(input_file, face_file)
        merged_face_file = os.path.join(self.tmp_test_sample_dir, name_wo_ext + '_face_fake_merged' + ext)
        generate_fake_merged_image(face_file, merged_face_file)

        # generate depth image of face
        self.gen_depth_image(self.opt_face, self.model_face)

        # merge face into figure image
        figure_depth_path = os.path.join(self.figure_results_dir, 'figure_pix2pix', 'test_latest', 'images',
                                         name_wo_ext + '_fake_merged_fake_B.png')
        face_depth_path = os.path.join(self.figure_results_dir, 'face_pix2pix', 'test_latest', 'images',
                                         name_wo_ext + '_fake_merged_face_fake_B.png')
        image = Image.open(figure_depth_path)
        face_image = Image.open(face_depth_path)
        image.paste(face_image, (top, left))
        image.save(os.path.join(self.tmp_results_dir, name_wo_ext+'_depth.png'))

    def gen_face_depth(self, input_file):
        pass

    def merge(self, total_image, face_image, face_position):
        pass

    def save_8bit_grayscale(self, rgb_image, output):
        pass

    def save_16bit_grayscale(self, rgb_image, output):
        pass

    def process_test_samples(self, output):
        pass

    def generate(self, input, output):
        if not os.path.exists(output):
            os.makedirs(output)
        if os.path.isfile(input):
            #
            self.gen_image_depth(input)
        elif os.path.isdir(input):
            pass
        else:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of DepthImageGenerator')
    parser.add_argument("input", type=str, help="input image file/dir")
    parser.add_argument("-o", "--output", type=str, default='generated', help="output image dir")
    args = parser.parse_args()

    gen = DepthGenerator()
    gen.generate(args.input, args.output)
