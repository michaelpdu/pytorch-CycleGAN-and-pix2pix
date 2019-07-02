import os
import argparse
from PIL import Image
import numpy as np

class ImageHelper:
    def __init__(self):
        pass

    def generate_merged_images(self, root_dir, output_dir):
        for filename in os.listdir(root_dir):
            name_wo_ext, ext = os.path.splitext(filename)
            if name_wo_ext.endswith('_color'):
                print(filename)
                filename_wo_surfix = name_wo_ext[0:-6]
                depth_image_name = filename_wo_surfix + '_depth' + '.png'

                image_rgb = Image.open(os.path.join(root_dir, filename))
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

    def convert_16_to_8(self, input, output):
        img16 = Image.open(input)
        ratio = np.amax(img16) / 256
        img8 = (img16 / ratio).astype('uint8')
        image = Image.fromarray(img8)
        image.save(output)
        image.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of ImageHelper')
    parser.add_argument("-i", "--input", type=str, help="input image dir")
    parser.add_argument("-o", "--output", type=str, help="output image dir")
    parser.add_argument("-m", "--merge", action='store_true', help="merge two images")
    parser.add_argument("--c16to8", action='store_true', help="convert 16-bit grayscale to 8-bit grayscale")
    args = parser.parse_args()

    if args.input:
        helper = ImageHelper()
        if args.merge:
            if not os.path.exists(args.output):
                os.mkdir(args.output)
            helper.generate_merged_images(args.input, args.output)
        elif args.c16to8:
            helper.convert_16_to_8(args.input, args.output)
    else:
        parser.print_help()