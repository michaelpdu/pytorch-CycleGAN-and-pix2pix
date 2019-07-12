import os
import argparse
from PIL import Image
import cv2
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_helper import get_face_rect

def crop_square(image_path, square_path):
    image = Image.open(image_path)
    ow, oh = image.size
    cut = abs(ow - oh) / 2
    cropped_image = image.crop((cut, 0, oh + cut, oh))
    cropped_image.save(square_path)

def generate_fake_merged_image(image_path, fake_merged_path):
    image = Image.open(image_path)
    ow, oh = image.size
    cut = abs(ow - oh) / 2
    cropped_image = image.crop((cut, 0, oh + cut, oh))
    new_image = Image.new('RGB', (oh * 2, oh))
    new_image.paste(cropped_image, (0, 0))
    new_image.save(fake_merged_path)

class ImageHelper:
    def __init__(self):
        self.avg_ratio = 3.6
        self.total_ratio = 0
        self.count = 0
        self.extra = 20

    def generate_fake_merged_images(self, root_dir, output_dir):
        for filename in os.listdir(root_dir):
            name_wo_ext, ext = os.path.splitext(filename)
            if ext != '.jpg' and ext != '.png':
                continue
            input_file = os.path.join(root_dir, filename)
            output_file = os.path.join(output_dir, name_wo_ext + '_fake_merged' + ext)
            generate_fake_merged_image(input_file, output_file)

    def generate_merged_face_images(self, root_dir, output_dir):
        for filename in os.listdir(root_dir):
            name_wo_ext, ext = os.path.splitext(filename)
            if name_wo_ext.endswith('_color'):
                print(filename)
                filename_wo_surfix = name_wo_ext[0:-6]
                depth_image_name = filename_wo_surfix + '_depth_8bit' + '.png'
                # depth_image_name = filename_wo_surfix + '_depth.raw.bmp'

                image_path = os.path.join(root_dir, filename)
                rect = get_face_rect(image_path)
                if rect is None:
                    continue
                (top, right, bottom, left) = rect
                rect = (left - self.extra, top - self.extra, right + self.extra, bottom + self.extra)
                image = Image.open(image_path)
                face_image = image.crop(rect)
                wa, ha = face_image.size
                image_depth = Image.open(os.path.join(root_dir, depth_image_name))
                face_image_depth = image_depth.crop(rect)
                wb, hb = face_image_depth.size

                # assert (wa == wb and ha == hb)
                new_image = Image.new('RGB', (wa+wb, ha))
                new_image.paste(face_image, (0, 0))
                new_image.paste(face_image_depth, (wa, 0))

                new_image.save(os.path.join(output_dir, filename_wo_surfix+'_merged_face'+'.jpg'))

    def generate_merged_images(self, root_dir, output_dir):
        for filename in os.listdir(root_dir):
            name_wo_ext, ext = os.path.splitext(filename)
            if name_wo_ext.endswith('_color'):
                print(filename)
                filename_wo_surfix = name_wo_ext[0:-6]
                # depth_image_name = filename_wo_surfix + '_depth_8bit' + '.png'
                depth_image_name = filename_wo_surfix + '_depth.raw.bmp'

                image_path = os.path.join(root_dir, filename)
                (top, right, bottom, left) = get_face_rect(image_path)

                image_rgb = Image.open(image_path)
                wa, ha = image_rgb.size
                image_depth = Image.open(os.path.join(root_dir, depth_image_name))
                wb, hb = image_depth.size

                assert (wa == wb and ha == hb)
                cut = (wa - ha) / 2

                new_image_rgb = image_rgb.crop((cut, 0, ha+cut, ha))
                new_image_depth = image_depth.crop((cut, 0, hb+cut, hb))

                new_image = Image.new('RGB', (ha*2, ha))
                new_image.paste(new_image_rgb, (0, 0))
                new_image.paste(new_image_depth, (ha, 0))

                new_image.save(os.path.join(output_dir, filename_wo_surfix+'_merge'+'.jpg'))

    def retrieve_grayscale_image(self, input, output):
        image = Image.open(input)
        width, height = image.size
        new_width = width/2
        gray_image_rgb = image.crop((new_width, 0, width, height))
        gray_image = gray_image_rgb.convert("LA")
        gray_image.save(output)

    def grayscale_16to8(self, input, output):
        img16 = Image.open(input)
        # print('max value in img16: {}'.format(np.amax(img16)))
        ratio = np.amax(img16) / 256
        self.count += 1
        self.total_ratio += ratio
        print('ratio:', ratio)
        img8 = (img16 / ratio).astype('uint8')
        image = Image.fromarray(img8)
        image.save(output)

    def grayscale_8to16(self, input, output):
        img8 = Image.open(input)
        print('max value in img8: {}'.format(np.amax(img8)))
        img16 = (np.array(img8)*self.avg_ratio).astype('uint16')
        print('max value in img16: {}'.format(np.amax(img16)))
        image = Image.fromarray(img16)
        image.save(output)

    def rgb888_to_8bit_grayscale(self, input, output):
        image = Image.open(input).convert('L')
        arr = np.array(image)
        arr = (255 - arr / 2)
        arr[arr > 245] = 0
        img8 = arr.astype('uint8')
        cv2.imwrite(output, img8)

    def rgb888_to_16bit_grayscale(self, input, output):
        image = Image.open(input).convert('L')
        arr = np.array(image)
        arr = (255 - arr / 2)
        arr[arr > 245] = 0
        img16 = (np.array(arr) * self.avg_ratio).astype('uint16')
        cv2.imwrite(output, img16)

    def batch_grayscale_conversion(self, input_dir, output_dir):
        for root, dirs, files in os.walk(input_dir):
            for name in files:
                name_wo_ext, ext = os.path.splitext(name)
                if name_wo_ext.endswith('_depth'):
                    input_file = os.path.join(root, name)
                    output_file = os.path.join(output_dir, name_wo_ext+'_8bit'+ext)
                    self.grayscale_16to8(input_file, output_file)
        print('average ratio:', self.total_ratio/self.count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of ImageHelper')
    parser.add_argument("-i", "--input", type=str, help="input image dir")
    parser.add_argument("-o", "--output", type=str, help="output image dir")
    parser.add_argument("-g", "--generate", action='store_true', help="generate fake merged images")
    parser.add_argument("-m", "--merge", action='store_true', help="merge two images")
    parser.add_argument("-mf", "--merge_face", action='store_true', help="merge face in two images")
    parser.add_argument("-r", "--retrieve", action='store_true', help="retrieve grayscale image")
    parser.add_argument("-c", "--convert", type=str, \
                        help="convert image type, gs16to8|gs8to16|rgb888to8bit|rgb888to16bit")
    args = parser.parse_args()

    if args.input:
        helper = ImageHelper()
        if args.merge:
            if not os.path.exists(args.output):
                os.mkdir(args.output)
            helper.generate_merged_images(args.input, args.output)
        elif args.merge_face:
            if not os.path.exists(args.output):
                os.mkdir(args.output)
            helper.generate_merged_face_images(args.input, args.output)
        elif args.generate:
            if not os.path.exists(args.output):
                os.mkdir(args.output)
            helper.generate_fake_merged_images(args.input, args.output)
        elif args.retrieve:
            helper.retrieve_grayscale_image(args.input, args.output)
        elif args.convert:
            if os.path.isfile(args.input):
                if args.convert == 'gs16to8':
                    helper.grayscale_16to8(args.input, args.output)
                elif args.convert == 'gs8to16':
                    helper.grayscale_8to16(args.input, args.output)
                elif args.convert == 'rgb888to8bit':
                    helper.rgb888_to_8bit_grayscale(args.input, args.output)
                else:
                    print('ERROR: Unimplemented Command!')
            elif os.path.isdir(args.input):
                if args.convert == 'gs16to8':
                    helper.batch_grayscale_conversion(args.input, args.output)
                else:
                    print('ERROR: Unimplemented Command!')
            else:
                parser.print_help()
        else:
            parser.print_help()
    else:
        parser.print_help()
