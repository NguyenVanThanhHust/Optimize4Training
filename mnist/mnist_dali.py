# https://gist.github.com/irustandi/3d180ed3ec9d7ff4e73d3fdbd67df3ca

from argparse import ArgumentParser

import numpy as np
from random import shuffle

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

class ExternalMNISTInputIterator(object):
    def __init__(self, mnist_ds, batch_size) -> None:
        self.batch_size = batch_size
        self.mnist_ds = mnist_ds
        self.indices = list(range(len(self.mnist_ds)))
        shuffle(self.indices)

    def __iter__(self, ):
        self.i = 0
        self.n = len(self.mnist_ds)
        return self
    
    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            index = self.indices[self.i]
            img, label = self.mnist_ds[index]
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
        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x


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
    dataset = MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    external_input_iterator_train = ExternalMNISTInputIterator(mnist_ds=mnist_train, batch_size=args.batch_size)
    external_input_iterator_val = ExternalMNISTInputIterator(mnist_ds=mnist_val, batch_size=args.batch_size)
    external_input_iterator_test = ExternalMNISTInputIterator(mnist_ds=mnist_test, batch_size=args.batch_size)

    pipe_train = ExternalSourcePipeline(external_iterator=external_input_iterator_train, batch_size=args.batch_size, num_threads=2, device_id=0)
    pipe_train.build()
    train_loader = DALIClassificationLoader(pipe_train, size=len(mnist_train), auto_reset=True, last_batch_policy=False)

    pipe_val = ExternalSourcePipeline(external_iterator=external_input_iterator_val, batch_size=args.batch_size, num_threads=2, device_id=0)
    pipe_val.build()
    val_loader = DALIClassificationLoader(pipe_val, size=len(mnist_val), auto_reset=True, last_batch_policy=False)

    pipe_test = ExternalSourcePipeline(external_iterator=external_input_iterator_test, batch_size=args.batch_size, num_threads=2, device_id=0)
    pipe_test.build()
    test_loader = DALIClassificationLoader(pipe_test, size=len(mnist_test), auto_reset=True, last_batch_policy=False)


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
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()

# python mnist_dali.py --learning_rate 0.001 --hidden_dim 128 --accelerator cuda

