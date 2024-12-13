# https://gist.github.com/irustandi/3d180ed3ec9d7ff4e73d3fdbd67df3ca

from argparse import ArgumentParser

import numpy as np
from random import shuffle

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST
from torchvision import transforms

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

from models import ResNet18

class ExternalCifarInputIterator(object):
    def __init__(self, cifar_ds, batch_size) -> None:
        self.batch_size = batch_size
        self.cifar_ds = cifar_ds
        self.indices = list(range(len(self.cifar_ds)))
        shuffle(self.indices)

    def __iter__(self, ):
        self.i = 0
        self.n = len(self.cifar_ds)
        return self
    
    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            index = self.indices[self.i]
            img, label = self.cifar_ds[index]
            batch.append(img.numpy())
            labels.append(np.array([label],dtype = np.uint8))
            self.i = (self.i+1)%self.n
        return (batch, labels)
    
class ExternalSourcePipeline(Pipeline):
    def __init__(self, external_iterator, batch_size, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=12)
        self.source = ops.ExternalSource(source=external_iterator, num_outputs=2)
    
    def define_graph(self):
        images, labels = self.source()
        return images, labels
        
class DALIClassificationLoader(DALIClassificationIterator):
    def __init__(
        self,
        pipelines,
        size=-1,
        reader_name=None,
        auto_reset=False,
        last_batch_policy=True,
        dynamic_shape=False,
        last_batch_padded=False,    
    ):
        super().__init__(pipelines, 
                         size, 
                         reader_name, 
                         auto_reset, 
                         last_batch_policy, 
                         dynamic_shape, 
                         last_batch_padded)

    def __len__(self):
        batch_count = self._size // (self._num_gpus * self.batch_size)
        last_batch = 1 if self._last_batch_policy else 0
        return batch_count + last_batch

class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim=128, lr=0.001):
        super().__init__()
        self.hparams.hhidden_dim = hidden_dim
        self.hparams.learning_rate = lr
        self.save_hyperparameters()
        self.net = ResNet18()

    def forward(self, x):
        output = self.net(x)
        return output


    def split_batch(self, batch):
        return batch[0]['data'], batch[0]['label'].squeeze().long()
    
    def training_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log('valid_loss', loss)
        
    
    def test_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log('test_loss', loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    print(f"args: {args}")
    # ------------
    # data
    # ------------
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    external_input_iterator_train = ExternalCifarInputIterator(cifar_ds=trainset, batch_size=args.batch_size)
    external_input_iterator_val = ExternalCifarInputIterator(cifar_ds=trainset, batch_size=args.batch_size)
    external_input_iterator_test = ExternalCifarInputIterator(cifar_ds=testset, batch_size=args.batch_size)

    pipe_train = ExternalSourcePipeline(external_iterator=external_input_iterator_train, batch_size=args.batch_size, num_threads=2, device_id=0)
    pipe_train.build()
    train_loader = DALIClassificationLoader(pipe_train, size=len(trainset), auto_reset=True, last_batch_policy=False)

    pipe_val = ExternalSourcePipeline(external_iterator=external_input_iterator_val, batch_size=args.batch_size, num_threads=2, device_id=0)
    pipe_val.build()
    val_loader = DALIClassificationLoader(pipe_val, size=len(trainset), auto_reset=True, last_batch_policy=False)

    pipe_test = ExternalSourcePipeline(external_iterator=external_input_iterator_test, batch_size=args.batch_size, num_threads=2, device_id=0)
    pipe_test.build()
    test_loader = DALIClassificationLoader(pipe_test, size=len(testset), auto_reset=True, last_batch_policy=False)


    # ------------
    # model
    # ------------
    model = LitClassifier(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()

# python cifar_dali.py --learning_rate 0.001 --hidden_dim 128 --accelerator cuda