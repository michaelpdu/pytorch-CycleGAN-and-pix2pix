import argparse
from PIL import Image, ImageEnhance
import numpy as np
import cv2

# def grayscale_enhance(self, input, output):
#     img8 = Image.open(input)
#     print('max value in img8: {}'.format(np.amax(img8)))
#     # img16 = (np.array(img8) * self.avg_ratio).astype('uint16')
#     # print('max value in img16: {}'.format(np.amax(img16)))
#     # image = Image.fromarray(img16)
#     # image.save(output)
#
#     enhancer = ImageEnhance.Contrast(img8)
#     enhanced_im = enhancer.enhance(4.0)
#
# grayscale_enhance()

def rgb888_to_8bit_grayscale(input, output):
    image = Image.open(input).convert('L')
    arr = np.array(image)
    # width, height = image.size
    arr = (255 - arr/2)
    arr[arr > 245] = 0
    # new_image = Image.fromarray(arr)
    # new_image.convert('')
    # grayscale_image.save(output)
    img8 = arr.astype('uint8')
    cv2.imwrite(output, img8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of Test')
    parser.add_argument("input", type=str, help="input image ")
    parser.add_argument("output", type=str, help="output image ")
    args = parser.parse_args()

    rgb888_to_8bit_grayscale(args.input, args.output)
