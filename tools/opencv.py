import cv2
import argparse
import numpy as np

def gray2bgr565(input_file, output_file):
    img = np.fromfile(input_file, dtype=np.uint16)
    img = img.reshape(480, 640)
    # img = cv2.imread(input_file, cv2.IMREAD_ANYDEPTH)
    ratio = np.amax(img) / 256
    img8 = (img / ratio).astype('uint8')
    img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(output_file, img8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of ImageHelper')
    parser.add_argument("-i", "--input", type=str, help="input image dir")
    parser.add_argument("-o", "--output", type=str, help="output image dir")
    args = parser.parse_args()

    if args.input:
        gray2bgr565(args.input, args.output)
    else:
        parser.print_help()