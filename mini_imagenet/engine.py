import os
from typing import Iterable

import torch
from torch.utils.tensorboard import SummaryWriter

from evalutor import ClassificationEvalutor
import utils.misc as utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler, device: torch.device,
                    writer: SummaryWriter, epoch: int):
    model.train()
    criterion.train()

    num_sample_per_epoch = len(data_loader)
    for idx, (images, targets) in enumerate(data_loader):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss_value = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss_value, idx + num_sample_per_epoch*epoch)

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
                    device: torch.device, writer: SummaryWriter, epoch: int):
    model.eval()
    criterion.eval()
    evaluator = ClassificationEvalutor()

    num_sample_per_epoch = len(data_loader)
    for idx, (images, targets) in enumerate(data_loader):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss_value = loss.item()

        acc = evaluator.add(outputs, targets)
        writer.add_scalar("Loss/val", loss_value, idx + num_sample_per_epoch*epoch)
        writer.add_scalar("Loss/acc", acc, idx + num_sample_per_epoch*epoch)
    return evaluator.summerize()