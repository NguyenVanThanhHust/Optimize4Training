import os
from os.path import join
import argparse
import random
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    input = args.input
    output = args.output
    os.makedirs(output, exist_ok=True)
    
    folders = next(os.walk(input))[1]
    train_folder = join(output, "train")
    val_folder = join(output, "val")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    for folder in folders:
        old_folder_path = join(input, folder)
        new_train_folder_path = join(train_folder, folder)
        new_val_folder_path = join(val_folder, folder)
        os.makedirs(new_train_folder_path, exist_ok=True)
        os.makedirs(new_val_folder_path, exist_ok=True)
        files = next(os.walk(old_folder_path))[2]
        for f in files:
            old_file_path = join(old_folder_path, f)
            if random.random() > 0.2:
                new_file_path = join(new_train_folder_path, f)
            else:
                new_file_path = join(new_val_folder_path, f)
            shutil.copy(old_file_path, new_file_path)
            print(f"copy from {old_file_path} to {new_file_path}")
                