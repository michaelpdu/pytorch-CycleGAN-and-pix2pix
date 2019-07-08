import argparse
import face_recognition
from PIL import Image

GAP = 20

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of ImageHelper')
    parser.add_argument("-i", "--input", type=str, help="input image file")
    parser.add_argument("-o", "--output", type=str, default='output.jpg', help="output image file")
    args = parser.parse_args()
    if args.input:
        save_face_image(args.input, args.output)
    else:
        parser.print_help()