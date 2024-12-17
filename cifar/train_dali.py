# https://gist.github.com/irustandi/3d180ed3ec9d7ff4e73d3fdbd67df3ca

from argparse import ArgumentParser

import numpy as np
from random import shuffle

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import transforms

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models import ResNet18

from torchmetrics.classification import Accuracy, MulticlassAccuracy


class ExternalCifarInputIterator(object):
    def __init__(self, cifar_ds, batch_size) -> None:
        self.batch_size = batch_size
        self.cifar_ds = cifar_ds
        self.indices = list(range(len(self.cifar_ds)))
        shuffle(self.indices)

    def __iter__(
        self,
    ):
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
            labels.append(np.array([label], dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)


class ExternalSourcePipeline(Pipeline):
    def __init__(self, external_iterator, batch_size, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(
            batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=12
        )
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
        super().__init__(
            pipelines,
            size,
            reader_name,
            auto_reset,
            last_batch_policy,
            dynamic_shape,
            last_batch_padded,
        )

    def __len__(self):
        batch_count = self._size // (self._num_gpus * self.batch_size)
        last_batch = 1 if self._last_batch_policy else 0
        return batch_count + last_batch


class LitClassifier(pl.LightningModule):
    def __init__(self, lr=0.001, num_classes=10):
        super().__init__()
        self.hparams.learning_rate = lr
        self.save_hyperparameters()
        self.net = ResNet18()
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = Accuracy(num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

    def split_batch(self, batch):
        return batch[0]["data"], batch[0]["label"].squeeze().long()

    def training_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_pred = self(x)
        train_loss = F.cross_entropy(y_pred, y)
        self.log("train_loss", train_loss)
        self.train_accuracy(y_pred, y)
        self.log("train_acc_step", self.train_accuracy)
        return train_loss

    def on_training_epoch_end(self) -> None:
        # log epoch metric
        self.log("train_acc_epoch", self.train_accuracy)

    def validation_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_pred = self(x)
        valid_loss = F.cross_entropy(y_pred, y)
        self.log("valid_loss", valid_loss)
        self.val_accuracy(y_pred, y)
        self.log("val_acc_step", self.val_accuracy)
        return valid_loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_acc_epoch", self.val_accuracy)

    def test_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # ------------
    # data
    # ------------

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    cifar_trainval = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    cifar_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    cifar_train, cifar_val = random_split(cifar_trainval, [45000, 5000])

    external_input_iterator_train = ExternalCifarInputIterator(
        cifar_ds=cifar_train, batch_size=args.batch_size
    )
    external_input_iterator_val = ExternalCifarInputIterator(
        cifar_ds=cifar_val, batch_size=args.batch_size
    )
    external_input_iterator_test = ExternalCifarInputIterator(
        cifar_ds=cifar_test, batch_size=args.batch_size
    )

    pipe_train = ExternalSourcePipeline(
        external_iterator=external_input_iterator_train,
        batch_size=args.batch_size,
        num_threads=2,
        device_id=0,
    )
    pipe_train.build()
    train_loader = DALIClassificationLoader(
        pipe_train, size=len(cifar_train), auto_reset=True, last_batch_policy=False
    )

    pipe_val = ExternalSourcePipeline(
        external_iterator=external_input_iterator_val,
        batch_size=args.batch_size,
        num_threads=2,
        device_id=0,
    )
    pipe_val.build()
    val_loader = DALIClassificationLoader(
        pipe_val, size=len(cifar_val), auto_reset=True, last_batch_policy=False
    )

    pipe_test = ExternalSourcePipeline(
        external_iterator=external_input_iterator_test,
        batch_size=args.batch_size,
        num_threads=2,
        device_id=0,
    )
    pipe_test.build()
    test_loader = DALIClassificationLoader(
        pipe_test, size=len(cifar_test), auto_reset=True, last_batch_policy=False
    )

    # ------------
    # model
    # ------------
    model = LitClassifier(lr=args.learning_rate, num_classes=10)

    # ------------
    # training
    # ------------
    early_stop_callback = EarlyStopping(
        monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max"
    )
    model_checkpoint = ModelCheckpoint(
        monitor="val_accuracy",
        dirpath="./ckpts",  # f"{args.checkpoint_dir}/{args.save_model_name}",
        filename="model_ckpt",
        save_top_k=2,
        mode="max",
        save_last=True,
    )

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[model_checkpoint, early_stop_callback], detect_anomaly=True
    )
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()

# python train_dali.py --learning_rate 0.001  --accelerator cuda
