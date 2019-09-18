import argparse
import face_recognition
from PIL import Image

GAP = 40

def get_face_rect(image_path):
    image_array = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image_array)
    print(face_locations)
    if len(face_locations) == 0:
        return None
    return face_locations[0]

def save_face_image(image_path, face_path):
    (top, right, bottom, left) = get_face_rect(image_path)
    print('Width: {}, Height: {}'.format(right-left, bottom-top))
    image = Image.open(image_path)
    face_image = image.crop((left-GAP, top-GAP, right+GAP, bottom+GAP))
    face_image.save(face_path)
    return (left-GAP, top-GAP, right+GAP, bottom+GAP)

def remove_gap(image_path, new_image_path):
    image = Image.open(image_path)
    ow, oh = image.size
    new_image = image.crop((GAP, GAP, oh-GAP, ow-GAP))
    new_image.save(new_image_path)

def gen_640_480_image(image_path, new_image_path):
    figure_image = Image.open(image_path)
    figure_image = figure_image.resize((480, 480))
    # prepare a new black image 640x480
    image = Image.new('RGB', (640, 480))
    image.paste(figure_image, (80, 0, 480+80, 480))
    image.save(new_image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of ImageHelper')
    parser.add_argument("-i", "--input", type=str, help="input image file")
    parser.add_argument("-o", "--output", type=str, default='output.jpg', help="output image file")
    parser.add_argument("-t", "--type", type=str, help="save_face|gen_640")
    args = parser.parse_args()
    if args.input:
        if args.type == 'save_face':
            save_face_image(args.input, args.output)
        elif args.type == 'gen_640':
            gen_640_480_image(args.input, args.output)
        else:
            parser.print_help()
    else:
        parser.print_help()