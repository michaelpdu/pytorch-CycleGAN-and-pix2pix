from mpl_toolkits import mplot3d
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt


def draw_image(image_path, type):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    mode = 'LA' if type == '16bit' else 'L'
    print('mode = ', mode)
    image = Image.open(image_path).convert(mode)
    arr = np.array(image)
    width, height = image.size
    value = image.getpixel((1, 1))
    print('value:', value)
    x = np.linspace(0, width, width, dtype=int)
    y = np.linspace(0, height, height, dtype=int)
    X, Y = np.meshgrid(x, y)
    if type == '8bit':
        arr = (255 - arr)
        arr[arr >= 245] = 0
    elif type == 'rgb888':
        pass
        # arr = (255 - arr / 2)
        # arr[arr >= 245] = 0
    elif type == '16bit':
        print('ERROR: Unsupported Command')
        # arr = arr/4
        # arr = (255 - arr)
        # arr[arr >= 245] = 0
    else:
        pass
    Z = arr
    ax.plot_wireframe(X, Y, Z, color='green')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='winter', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of ImageDepthDrawer')
    parser.add_argument("image_path", type=str, help="input image path")
    parser.add_argument("-t", "--type", type=str, help="image type, 8bit|16bit|rgb888|raw")
    args = parser.parse_args()

    if args.image_path:
        draw_image(args.image_path, args.type)
    else:
        parser.print_help()
