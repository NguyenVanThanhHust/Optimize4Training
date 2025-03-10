import torch.nn as nn
from .resnet18 import ResNet18, ResNet50

def build_model(args):
    model = ResNet18()
    criterion = nn.CrossEntropyLoss()
    return model, criterion