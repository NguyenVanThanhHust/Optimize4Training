import argparse

import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from imagenet_dataloader import MiniImageNetDataset, build_transform
from models import build_model
from engine import train_one_epoch, evaluate

def get_arg_parser():
    parser = argparse.ArgumentParser('Mini Imagenet', add_help=False)
    parser.add_argument("--config-file", type=str, default="configs/baseline.py")
    parser.add_argument("--data_dir", type=str, default="../../Datasets/mini_imagenet")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    return args

def main():
    args = get_arg_parser()

    model, criterion = build_model(args)
    device = torch.device(args.device)
    model.to(device)

    train_transforms = build_transform()
    dataset_train = MiniImageNetDataset(args.data_dir, ds_type="train", transform=train_transforms)
    dataset_val = MiniImageNetDataset(args.data_dir, ds_type="val", transform=train_transforms)

    train_dataloader = DataLoader(dataset_train, shuffle=False)
    val_dataloader = DataLoader(dataset_val, shuffle=False)

    optim = torch.optim.SGD(model.parameters())
    output_dir = "runs"
    writer = SummaryWriter(output_dir)

    epochs = args.epochs
    for i in range(epochs):
        # train_one_epoch(model, criterion, train_dataloader, optim, None, device, writer, i)
        evaluate(model, criterion, val_dataloader, device, writer, i)
    writer.flush()
    writer.close()
    
if __name__ == '__main__':
    main()