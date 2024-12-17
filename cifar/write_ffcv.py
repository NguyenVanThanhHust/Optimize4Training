from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
import torchvision

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField


def main():
    datasets = {
        "train": torchvision.datasets.CIFAR10("./data", train=True, download=True),
        "test": torchvision.datasets.CIFAR10("./data", train=False, download=True),
    }
    out_path = {
        "train": "./data/cifar10_train.beton",
        "test": "./data/cifar10_test.beton",
    }

    for name, ds in datasets.items():
        path = out_path[name]
        writer = DatasetWriter(path, {"image": RGBImageField(), "label": IntField()})
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    main()
