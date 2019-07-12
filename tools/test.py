import argparse
from PIL import Image, ImageEnhance
import numpy as np
import cv2

def grayscale_enhance(input, output):
    img8 = Image.open(input)
    print('max value in img8: {}'.format(np.amax(img8)))
    # img16 = (np.array(img8) * self.avg_ratio).astype('uint16')
    # print('max value in img16: {}'.format(np.amax(img16)))
    # image = Image.fromarray(img16)
    # image.save(output)
    img8.show('Grayscale 8bit')
    enhancer_contrast = ImageEnhance.Contrast(img8)
    img_contrast = enhancer_contrast.enhance(4.0)
    img_contrast.show('Contrast Enhancement (4.0)')
    enhancer_brightness = ImageEnhance.Brightness(img_contrast)
    img_brightness = enhancer_brightness.enhance(0.2)
    img_brightness.show('brightness enhancement')

def rgb888_to_8bit_grayscale(input, output):
    image = Image.open(input).convert('L')
    arr = np.array(image)
    arr = (255 - arr/2)
    arr[arr > 245] = 0
    img8 = arr.astype('uint8')
    cv2.imwrite(output, img8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of Test')
    parser.add_argument("input", type=str, help="input image ")
    parser.add_argument("output", type=str, help="output image ")
    args = parser.parse_args()

    # rgb888_to_8bit_grayscale(args.input, args.output)
    grayscale_enhance(args.input, args.output)
