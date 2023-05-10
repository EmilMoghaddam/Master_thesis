import warnings
import pandas as pd
import numpy as np
import cv2
import pytorch_lightning
import torch
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
import transformers
import time
import piexif

# pytorch and pytorch lightning
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core import datamodule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


# Vision transfoemer
from transformers import ViTForImageClassification
from transformers import ViTConfig
from transformers import ViTFeatureExtractor
from transformers import ViTImageProcessor

# Resnet
from transformers import ResNetForImageClassification, ResNetConfig

# BeIT
from transformers import AutoImageProcessor, BeitForImageClassification

#Convnext
from transformers import AutoImageProcessor, ConvNextForImageClassification

from transformers import pipeline  # What does this do?
from transformers import AutoImageProcessor  # Should match dimentions of image -- Need to make my own
from datasets import load_dataset


from torchvision import transforms
import torch.nn as nn
from torch.nn.functional import softmax

from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split

from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from getsun import SUN

pytorch_lightning.seed_everything(42)

class ViTClassifier(nn.Module):

    def __init__(self, num_labels=397):

        super().__init__()

        self.num_labels = num_labels
        self.loss_fn = nn.CrossEntropyLoss()

        configuration = ViTConfig(image_size=(224, 448))
        self.model = ViTForImageClassification(configuration)
        self.model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, num_labels))  # nn.Softmax(dim = -1)

    def forward(self, pixel_values, labels):

        logits = self.model(pixel_values=pixel_values)

        loss = None
        if labels is not None:

            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return logits, loss.item()
        else:
            return logits, None


class TorchDataset(Dataset):

    def __init__(self, data=SUN, mean = [0.485, 0.456, 0.406], size = (224, 224), std = [0.229, 0.224, 0.225], stage="train"):

        self.data = data
        #size = (224, 448)

        train_transform = transforms.Compose([transforms.Resize(size),
                                              transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

        val_transform = transforms.Compose([transforms.Resize(size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])

        if stage == "train":
            self.transform = train_transform
        elif stage == "validation":
            self.transform = val_transform

        self.target_transform = transforms.ToTensor()

    def __len__(self):  # Note that counts for all data! might cause error
        return len(self.data)

    def __getitem__(self, idx):

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            image, label = self.data[idx]

            image = self.transform(image)

            label = torch.tensor(label, dtype=torch.int64)

        return image, label


class LightningWrapper(LightningDataModule):

    def __init__(self, data=SUN, batch_size=16, train_workers=12, val_workers=12, mean = [0.485, 0.456, 0.406], size = (224, 224), std = [0.229, 0.224, 0.225]):

        super().__init__()

        self.data = data
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers

        targets = self.data._labels
        train_idx, valid_idx = train_test_split(np.arange(len(self.data)), test_size=0.2, shuffle=True, stratify=targets)

        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        self.valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        self.SUN_train = TorchDataset(data=self.data, mean = mean, size= size, std = std, stage="train")
        self.SUN_val = TorchDataset(data=self.data, mean = mean, size= size, std = std, stage="validation")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.SUN_train, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.train_workers)  # NUM_WORKERS

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.SUN_val, batch_size=self.batch_size, sampler=self.valid_sampler, num_workers=self.val_workers)


class VitLightningModule(pl.LightningModule):
    def __init__(self, num_labels=397, lr=1.0e-03):
        super().__init__()

        self.save_hyperparameters()

        self.loss_function = nn.CrossEntropyLoss()

        #configuration = ViTConfig(image_size=(224, 448))
        #self.model = ViTForImageClassification(configuration)

        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

        self.model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, num_labels))

        self.accuracy_top_1 = MulticlassAccuracy(num_classes=self.hparams.num_labels, top_k=1, average="micro")
        self.accuracy_top_5 = MulticlassAccuracy(num_classes=self.hparams.num_labels, top_k=5, average="micro")


    def forward(self, X):
        logits = self.model(X)["logits"]
        return logits

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        probs = softmax(logits, dim=1)

        loss = self.loss_function(logits, y)
        self.log("train_loss", loss, on_epoch=True)

        acc1 = self.accuracy_top_1(probs, y)
        self.log("train_accuracy_1", acc1, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        probs = softmax(logits, dim=1)

        loss = self.loss_function(logits, y)
        acc1 = self.accuracy_top_1(probs, y)
        acc5 = self.accuracy_top_5(probs, y)

        self.log("Validation_loss", loss, on_epoch=True)
        self.log("Validation_accuracy_1", acc1, on_epoch=True, prog_bar=True)
        self.log("Validation_accuracy_5", acc5, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class ResNetLightningModule(pl.LightningModule):
    def __init__(self, num_labels=397, lr=1.0e-03):
        super().__init__()

        self.save_hyperparameters()

        self.loss_function = nn.CrossEntropyLoss()

        config = ResNetConfig(image_size=(224, 448), downsample_in_first_stage=False)
        self.model = ResNetForImageClassification(config)

        #self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Flatten(start_dim=1, end_dim=-1), nn.Linear(2048, num_labels))

        self.accuracy_top_1 = MulticlassAccuracy(num_classes=self.hparams.num_labels, top_k=1, average="micro")
        self.accuracy_top_5 = MulticlassAccuracy(num_classes=self.hparams.num_labels, top_k=5, average="micro")

    def forward(self, X):
        logits = self.model(X).logits
        return logits

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        probs = softmax(logits, dim=1)

        loss = self.loss_function(logits, y)
        self.log("train_loss", loss, on_epoch=True)

        acc1 = self.accuracy_top_1(probs, y)
        self.log("train_accuracy_1", acc1, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        probs = softmax(logits, dim=1)

        loss = self.loss_function(logits, y)
        acc1 = self.accuracy_top_1(probs, y)
        acc5 = self.accuracy_top_5(probs, y)

        self.log("Validation_loss", loss, on_epoch=True)
        self.log("Validation_accuracy_1", acc1, on_epoch=True, prog_bar=True)
        self.log("Validation_accuracy_5", acc5, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class BeITLightningModule(pl.LightningModule):
    def __init__(self, num_labels=397, lr=2.0e-05):
        super().__init__()

        self.save_hyperparameters()

        self.loss_function = nn.CrossEntropyLoss()

        self.model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224",
                                                                ignore_mismatched_sizes=True,
                                                                label2id = label2id,
                                                                id2label = id2label)

        #print(self.model.classifier)
        #self.model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, num_labels, bias=True))

        self.accuracy_top_1 = MulticlassAccuracy(num_classes=self.hparams.num_labels, top_k=1, average="micro")
        self.accuracy_top_5 = MulticlassAccuracy(num_classes=self.hparams.num_labels, top_k=5, average="micro")

    def forward(self, X):
        logits = self.model(X)["logits"]
        return logits

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        probs = softmax(logits, dim=1)

        loss = self.loss_function(logits, y)
        self.log("train_loss", loss, on_epoch=True)

        acc1 = self.accuracy_top_1(probs, y)
        self.log("train_accuracy_1", acc1, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        probs = softmax(logits, dim=1)

        loss = self.loss_function(logits, y)
        acc1 = self.accuracy_top_1(probs, y)
        acc5 = self.accuracy_top_5(probs, y)

        self.log("Validation_loss", loss, on_epoch=True)
        self.log("Validation_accuracy_1", acc1, on_epoch=True, prog_bar=True)
        self.log("Validation_accuracy_5", acc5, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def stats(self):
        p = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
        mean, std, size = p.image_mean, p.image_std, (p.size['height'], p.size['width'])
        return (mean, std, size)


class ConvNextLightningModule(pl.LightningModule):
    def __init__(self, num_labels=397, lr=2.0e-05):
        super().__init__()

        self.save_hyperparameters()

        self.loss_function = nn.CrossEntropyLoss()

        self.model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224",
                                                                ignore_mismatched_sizes=True,
                                                                label2id = label2id,
                                                                id2label = id2label)

        self.accuracy_top_1 = MulticlassAccuracy(num_classes=self.hparams.num_labels, top_k=1, average="micro")
        self.accuracy_top_5 = MulticlassAccuracy(num_classes=self.hparams.num_labels, top_k=5, average="micro")

    def forward(self, X):
        logits = self.model(X)["logits"]
        return logits

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        probs = softmax(logits, dim=1)

        loss = self.loss_function(logits, y)
        self.log("train_loss", loss, on_epoch=True)

        acc1 = self.accuracy_top_1(probs, y)
        self.log("train_accuracy_1", acc1, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        probs = softmax(logits, dim=1)

        loss = self.loss_function(logits, y)
        acc1 = self.accuracy_top_1(probs, y)
        acc5 = self.accuracy_top_5(probs, y)

        self.log("Validation_loss", loss, on_epoch=True)
        self.log("Validation_accuracy_1", acc1, on_epoch=True, prog_bar=True)
        self.log("Validation_accuracy_5", acc5, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def stats(self):
        p = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
        mean, std, size = p.image_mean, p.image_std, (p.size['shortest_edge'], p.size['shortest_edge'])
        return (mean, std, size)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print(torch.cuda.device_count())

#root = "http://groups.csail.mit.edu/vision/SUN/"
#SUN = torchvision.datasets.SUN397(root=root, transform=None, target_transform=None, download=True)

id2label = {id:label for id, label in enumerate(SUN.classes)}
label2id = {label:id for id,label in id2label.items()}

#Models
###################

#beit = BeITLightningModule()
#mean, std, size = beit.stats()
#dataModule = LightningWrapper(mean = mean, std = std, size = size)

#vit = VitLightningModule()

#res = ResNetLightningModule()

next = ConvNextLightningModule()
mean, std, size = next.stats()
dataModule = LightningWrapper(mean = mean, std = std, size = size)


checkpoint_callback_acc = ModelCheckpoint(dirpath="/home/aisl/PycharmProjects/pythonProject/next_acc_checkpoint_pretrained", monitor="Validation_accuracy_1", mode="max")
checkpoint_callback_val = ModelCheckpoint(dirpath="/home/aisl/PycharmProjects/pythonProject/next_val_checkpoint_pretrained", monitor="Validation_loss")

# auto_scale_batch_size = None

trainer = Trainer(
    accelerator="auto",
    fast_dev_run=False,
    overfit_batches=False,
    auto_scale_batch_size=False,
    callbacks=[checkpoint_callback_acc, checkpoint_callback_val],
    track_grad_norm=2,
    devices=1 if torch.cuda.is_available() else None,   # GPU colab command
    #precision=16,
    auto_lr_find=True,
    gradient_clip_val=2,
    detect_anomaly=False,
    gradient_clip_algorithm="norm",
    max_epochs=100,
)

#print(torch.cuda.memory_summary(device=None, abbreviated=False))

# ResNet

#trainer.tune(res, dataModule)
#trainer.fit(res, dataModule)
#trainer.fit(res, dataModule, ckpt_path="")
#trainer.validate(res, dataModule)


# Vision Transformer

#trainer.tune(vit, dataModule)
#trainer.fit(vit, dataModule)
#trainer.fit(vit, dataModule, ckpt_path="")
#trainer.validate(vit, dataModule)


# BeIT
#trainer.fit(beit, dataModule)
#trainer.validate(beit, dataModule)


# ConvNext
#trainer.fit(next, dataModule)
#trainer.validate(next, dataModule)



