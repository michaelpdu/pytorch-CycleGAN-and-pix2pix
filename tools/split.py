import os
import shutil
import random
import argparse

class SplitHelper:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.train_dir = os.path.join(self.output_dir, 'train')
        self.val_dir = os.path.join(self.output_dir, 'val')

    def split_by_random(self, percent):
        if not os.path.exists(self.input_dir):
            raise('ERROR: input dir does not exists!')

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        if not os.path.exists(self.val_dir):
            os.makedirs(self.val_dir)

        filename_list = []
        for filename in os.listdir(self.input_dir):
            filename_list.append(filename)
        
        count = len(filename_list)
        random.shuffle(filename_list)

        p = int(count*percent)
        print('percent:', p)
        train_list = filename_list[0:p]
        val_list = filename_list[p:-1]
        
        for filename in train_list:
            shutil.copy2(os.path.join(self.input_dir, filename), self.train_dir)

        for filename in val_list:
            shutil.copy2(os.path.join(self.input_dir, filename), self.val_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages')
    parser.add_argument("input", type=str, help="input image dir")
    parser.add_argument("-o", "--output", type=str, default='output', help="output image dir")
    parser.add_argument("-p", "--percent", type=float, help="percent of train in total")
    args = parser.parse_args()
    split_helper = SplitHelper(args.input, args.output)
    split_helper.split_by_random(args.percent)
