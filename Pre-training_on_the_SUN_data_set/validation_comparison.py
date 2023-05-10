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

from pre import ViTClassifier, TorchDataset, LightningWrapper
from pre import label2id, id2label
from pre import ResNetLightningModule, BeITLightningModule, ConvNextLightningModule , VitLightningModule

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

beit = BeITLightningModule()
mean, std, size = beit.stats()
dataModule_Beit = LightningWrapper(mean = mean, std = std, size = size)

res = ResNetLightningModule()
dataModule_resnet = LightningWrapper(size = (224,248))


next = ConvNextLightningModule()
mean, std, size = beit.stats()
dataModule_next = LightningWrapper(mean = mean, std = std, size = size)


trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,   # GPU colab command
    logger=False,
)



print("pretrained_resnet")
trainer.validate(res, dataModule_resnet, ckpt_path="resnet_acc_checkpoint/epoch=47-step=261024.ckpt")
trainer.validate(res, dataModule_resnet, ckpt_path="resnet_val_checkpoint/epoch=5-step=32628.ckpt")

print("random_init_resnet")
trainer.validate(res, dataModule_resnet, ckpt_path="resnet_acc_checkpoint_untrained/epoch=87-step=478544.ckpt")
trainer.validate(res, dataModule_resnet, ckpt_path="resnet_val_checkpoint_untrained/epoch=32-step=179454.ckpt")

print("pretrained_beit")
trainer.validate(beit, dataModule_Beit, ckpt_path="beit_acc_checkpoint_pretrained/epoch=8-step=48942.ckpt")
trainer.validate(beit, dataModule_Beit, ckpt_path="beit_val_checkpoint_pretrained/epoch=4-step=27190.ckpt")

print("pretrained_next")
trainer.validate(next, dataModule_next, ckpt_path="next_acc_checkpoint_pretrained/epoch=8-step=48942.ckpt")
trainer.validate(next, dataModule_next, ckpt_path="next_val_checkpoint_pretrained/epoch=4-step=27190.ckpt")
