# Machine learning packages
import pandas as pd
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import sigmoid


import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision.utils import save_image

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core import datamodule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger

from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from pytorch_lightning.utilities.model_summary import ModelSummary

from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import F1Score
from torchmetrics.classification import MultilabelF1Score as F1
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelStatScores

import transformers
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTImageProcessor
from transformers import ResNetForImageClassification, ResNetConfig
from transformers import AutoImageProcessor, BeitForImageClassification  # BeIT
from transformers import ConvNextForImageClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Standard packages
import gc, os, subprocess, shutil, json, random, string, imageio, time, pickle

import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps

# for reproducebility
pl.seed_everything(42)

# Choose between gpu and cpu
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print(torch.cuda.device_count())

#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive


NUM_WORKERS = 4

ANNOTATION_FILE = "csv_files/new_photos.csv"
photo_csv_file = "csv_files/new_photos.csv"
camera_copy = r"camera/Camera_copy/"
segments_copy = r"segments/segments_copy/"
new_cam_csv_file = "csv_files/NewNavigationDataCameraExcel.csv"
new_seg_csv_file = "csv_files/NewNavigationDataSegmentsExcel.csv"

photos = pd.read_csv(photo_csv_file, index_col = 0)
#photos.photo_path = photos.photo_path.map({'/content/drive/MyDrive/Camera_copy': camera_copy, '/content/drive/MyDrive/segments_copy': segments_copy})

def read_video_dataset(fileName):
    df = pd.read_csv(fileName)
    df["FileName"].replace(' ', np.nan, inplace=True)
    df = df[df["FileName"].str.strip().astype(bool)]
    df.dropna(subset=["FileName"], inplace = True)
    df.sort_index(inplace = True)
    return df

def video_df():
    df_cam = read_video_dataset(new_cam_csv_file)
    df_seg = read_video_dataset(new_seg_csv_file)
    df_cam["photo_path"] = camera_copy
    df_seg["photo_path"] = segments_copy
    df = pd.concat([df_cam, df_seg], ignore_index=True, axis=0)
    df.at[len(df) - 1, 'Dynamic'] = 1
    df.at[len(df) - 1, 'Indoor'] = 1
    return df

df = video_df()
df['strat'] = df['Uneven'].astype(str) + df['Slope'].astype(str)
train_idx, test_idx = train_test_split(df.index, test_size=0.1, random_state =32, stratify=df['strat'])
train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state = 32, stratify=df.loc[train_idx]['strat'])
train_idx, val_idx, test_idx = list(train_idx), list(val_idx), list(test_idx)

CLASSES = ["Dynamic", "Outdoor", "Boundary", "Constrained", "Uneven", "Road", "Crowd", "Slope"]
p, t = transforms.ToPILImage(), transforms.ToTensor()

thresholds = [photos[k].sum()/len(photos) for k in CLASSES]

func = lambda idx : df.loc[idx][CLASSES].sum()

s1 = df.loc[train_idx][CLASSES].sum()
s2 = df.loc[val_idx][CLASSES].sum()
s3 = df.loc[test_idx][CLASSES].sum()

id2label = {id:label for id, label in enumerate(CLASSES)}
label2id = {label:id for id,label in id2label.items()}


count_vids = pd.DataFrame({'train': s1,'val':s2, 'test':s3, 'idx_col':s1.index})



# unresoled photoprocessor and photos
class TorchDataset(Dataset):
    def __init__(self, annotations_file, transform, idx_file, imageProcessor=None):

        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels[self.img_labels["original_index"].isin(idx_file)]

        self.transform = transform
        self.imageProcessor = imageProcessor

        #print("Transform type: ", type(self.transform))
        self.getidx = list(self.img_labels.index)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        new_idx = self.getidx[idx]
        p, n = photos.iloc[new_idx, [photos.columns.get_loc(c) for c in ['photo_path', 'FileName']]]
        img_path = os.path.join(p, n)

        # image = read_image(img_path).float() as tensor
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.imageProcessor != None:
            image = self.imageProcessor(image, return_tensors="pt")
            image = torch.squeeze(image['pixel_values'])  # if dim 3

        label = self.img_labels.iloc[idx]
        label = label[CLASSES]
        new_label = torch.as_tensor(label, dtype=torch.float)
        new_label = new_label

        return image, new_label, new_idx

    def getImage(self, idx):
        return self.__getitem__(idx)[0]

    def getLabel(self, idx):
        return self.__getitem__(idx)[1]


class LightningWrapper(LightningDataModule):

    def __init__(self, annotation_file, batch_size,
                size=(224, 224),
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]):

        super().__init__()

        self.annotation_file = annotation_file

        self.batch_size = batch_size

        auto_transform, test_transform = self.augment_and_transform(size = size, mean = mean, std = std)
        self.train_transform = auto_transform
        self.test_transform = test_transform

        self.collate_fn = None

    def prepare_data(self):
        self.train = TorchDataset(self.annotation_file, self.train_transform, train_idx)

        self.val = TorchDataset(self.annotation_file, self.test_transform, val_idx)
        self.test = TorchDataset(self.annotation_file, self.test_transform, test_idx)
        self.predict = TorchDataset(self.annotation_file, self.test_transform, list(df.index))

    def setup(self, stage=None):
        return  ###

    def augment_and_transform(self, size, mean, std):

        auto_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.AugMix(severity=3, mixture_width=3, chain_depth=- 1, alpha=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        return [auto_transform, test_transform]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=NUM_WORKERS, collate_fn=self.collate_fn,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=NUM_WORKERS, collate_fn=self.collate_fn,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=NUM_WORKERS, collate_fn=self.collate_fn,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=NUM_WORKERS, collate_fn=self.collate_fn,
                          shuffle=False)



class LitProgressBar(TQDMProgressBar):

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar

class RES(pl.LightningModule):

    def __init__(self, model_kwargs, lr, thresholds=8 * [0.5], pretrained = False, cp_path=None):
        super().__init__()

        self.save_hyperparameters()

        if pretrained:
            config = ResNetConfig.from_pretrained("microsoft/resnet-50")
        else:
            config = ResNetConfig(image_size=(224, 448), downsample_in_first_stage=False)

        self.model = ResNetForImageClassification(config)
        self.model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Flatten(start_dim=1, end_dim=-1), nn.Linear(2048, 8))

        state_dict = torch.load(cp_path)['state_dict']
        self.model.load_state_dict(state_dict=state_dict, strict=False)

        #self.example_input_array = torch.zeros_like(X)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

        # Creating thresholds
        print("Thresholds: ", thresholds)
        self.threshold = torch.Tensor(thresholds).to(device)

        self.F1 = F1(num_labels=8, average=None).to(device)

        self.F1_micro = F1(num_labels=6, average="micro").to(device)
        self.F1_macro = F1(num_labels=6, average="macro").to(device)
        self.F1_weighted = F1(num_labels=6, average="weighted").to(device)
        self.mlp = MultilabelPrecision(num_labels=8, average=None)
        self.mlr = MultilabelRecall(num_labels=8, average=None)

    def set_thresholds(self, thresholds):
        self.threshold = torch.Tensor(thresholds).to(device)

    def forward(self, x):
        logits = self.model(x)['logits']
        probs = sigmoid(logits)
        return logits, probs

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)

        return optimizer  # ], [lr_scheduler]

    def _logfn(self, logits, preds, labels, idxs, mode="train"):

        # Loss
        losses = torch.mean(self.loss_fn(logits, labels), dim=0)
        dict_loss = dict(zip(["%s loss " % mode + s for s in CLASSES], losses.tolist()))

        # accuracy
        accuracy = sum(torch.ge(preds, self.threshold) == labels) / len(labels)
        dict_accuracy = dict(zip(["%s accuracy " % mode + s for s in CLASSES], accuracy))

        # precission and recall
        precision = self.mlp(preds, labels)
        recall = self.mlr(preds, labels)
        dict_precision = dict(zip(["%s precision " % mode + s for s in CLASSES], precision))
        dict_recall = dict(zip(["%s recall " % mode + s for s in CLASSES], recall))

        # F1
        f1 = self.F1(preds, labels)
        dict_f1 = dict(zip(["%s f1 " % mode + s for s in CLASSES], f1))

        # Removing road and Crowd
        preds = torch.hstack((preds[:, :5], preds[:, [-1]]))
        labels = torch.hstack((labels[:, :5], labels[:, [-1]]))

        f1_micro = self.F1_micro(preds, labels)
        f1_macro = self.F1_macro(preds, labels)
        f1_weighted = self.F1_weighted(preds, labels)

        self.log("%s f1 micro" % mode, f1_micro)
        self.log("%s f1 macro" % mode, f1_macro)
        self.log("%s f1 weighted" % mode, f1_weighted)

        self.log_dict(dict_accuracy), self.log_dict(dict_loss), self.log_dict(dict_f1)
        self.log_dict(dict_precision), self.log_dict(dict_recall)

    def training_step(self, batch, batch_idx):

        imgs, labels, idxs = batch
        logits, preds = self.forward(imgs)
        loss = self.criterion(logits, labels)

        self._logfn(logits, preds, labels, imgs, mode="train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        imgs, labels, idxs = batch
        logits, preds = self.forward(imgs)
        loss = self.criterion(logits, labels)

        self._logfn(logits, preds, labels, imgs, mode="val")
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        imgs, labels, idxs = batch
        logits, preds = self.forward(imgs)

        self._logfn(logits, preds, labels, imgs, mode="test")

    def predict_step(self, batch, batch_idx):

        imgs, labels, idxs = batch
        logits, preds = self.forward(imgs)


        return (preds, labels)


class NeuralModel(pl.LightningModule):

    def __init__(self, model_kwargs, lr, thresholds=8 * [0.5]):
        super().__init__()

        self.save_hyperparameters()

        self.criterion = torch.nn.BCEWithLogitsLoss()


        # Creating thresholds
        print("Thresholds: ", thresholds)
        self.threshold = torch.Tensor(thresholds).to(device)

        # Epoch
        multidim_average = "global"
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
        self.F1 = F1(num_labels=8, average=None,  multidim_average = multidim_average).to(device)
        self.F1_micro = F1(num_labels=6, average="micro", multidim_average = multidim_average).to(device)
        self.F1_macro = F1(num_labels=6, average="macro", multidim_average = multidim_average).to(device)
        self.F1_weighted = F1(num_labels=6, average="weighted", multidim_average = multidim_average).to(device)
        self.mlp = MultilabelPrecision(num_labels=8, average=None, multidim_average = multidim_average).to(device)
        self.mlr = MultilabelRecall(num_labels=8, average=None, multidim_average = multidim_average).to(device)

        # Step
        self.loss_fn_step = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
        self.F1_micro_step = F1(num_labels=6, average="micro", multidim_average=multidim_average).to(device)

    def set_thresholds(self, thresholds):
        self.threshold = torch.Tensor(thresholds).to(device)

    def forward(self, x):
        logits = self.model(x)['logits']
        probs = sigmoid(logits)
        return logits, probs

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)

        return optimizer  # ], [lr_scheduler]

    def _logfn(self, logits, preds, labels, mode="train"):

        # Loss
        losses = torch.mean(self.loss_fn(logits, labels), dim=0)
        dict_loss = dict(zip(["%s loss " % mode + s for s in CLASSES], losses.tolist()))

        # accuracy
        accuracy = sum(torch.ge(preds, self.threshold) == labels) / len(labels)
        dict_accuracy = dict(zip(["%s accuracy " % mode + s for s in CLASSES], accuracy))

        # precission and recall
        precision = self.mlp(preds, labels)
        recall = self.mlr(preds, labels)
        dict_precision = dict(zip(["%s precision " % mode + s for s in CLASSES], precision))
        dict_recall = dict(zip(["%s recall " % mode + s for s in CLASSES], recall))

        # F1
        f1 = self.F1(preds, labels)
        dict_f1 = dict(zip(["%s f1 " % mode + s for s in CLASSES], f1))

        # Removing road and Crowd
        preds = torch.hstack((preds[:, :5], preds[:, [-1]]))
        labels = torch.hstack((labels[:, :5], labels[:, [-1]]))

        f1_micro = self.F1_micro(preds, labels)
        f1_macro = self.F1_macro(preds, labels)
        f1_weighted = self.F1_weighted(preds, labels)

        self.log("%s f1 micro" % mode, f1_micro, on_step = False, on_epoch = True)
        self.log("%s f1 macro" % mode, f1_macro, on_step = False, on_epoch = True)
        self.log("%s f1 weighted" % mode, f1_weighted, on_step = False, on_epoch = True)

        self.log_dict(dict_accuracy, on_step = False, on_epoch = True)
        self.log_dict(dict_loss, on_step = False, on_epoch = True)
        self.log_dict(dict_f1, on_step = False, on_epoch = True)
        self.log_dict(dict_precision, on_step = False, on_epoch = True)
        self.log_dict(dict_recall, on_step = False, on_epoch = True)

    def step_logfn(self, logits, preds, labels, mode="train"):

        # Removing road and Crowd
        preds = torch.hstack((preds[:, :5], preds[:, [-1]]))
        labels = torch.hstack((labels[:, :5], labels[:, [-1]]))
        f1_micro = self.F1_micro_step(preds, labels)
        self.log("%s f1 micro" % mode, f1_micro)

    def training_step(self, batch, batch_idx):

        imgs, labels, idxs = batch
        logits, preds = self.forward(imgs)
        loss = self.criterion(logits, labels)

        self.log("train step loss", loss)
        self.step_logfn(logits, preds, labels, mode="train step")

        return {'loss': loss, 'logits': logits, 'preds': preds, 'labels': labels}

    def validation_step(self, batch, batch_idx):

        imgs, labels, idxs = batch
        logits, preds = self.forward(imgs)
        loss = self.criterion(logits, labels)

        self.log("val step loss", loss, prog_bar="True")
        self.step_logfn(logits, preds, labels, mode="val step")

        return {'loss': loss, 'logits': logits, 'preds': preds, 'labels': labels}

    def test_step(self, batch, batch_idx):
        imgs, labels, idxs = batch
        logits, preds = self.forward(imgs)
        loss = self.criterion(logits, labels)

        self.log("val step loss", loss)
        self.step_logfn(logits, preds, labels, mode="test step")

        return {'loss': loss, 'logits': logits, 'preds': preds, 'labels': labels}

    def training_epoch_end(self, outputs):

        logits = torch.vstack([x['logits'] for x in outputs])
        preds = torch.vstack([x['preds'] for x in outputs])
        labels = torch.vstack([x['labels'] for x in outputs])
        loss = torch.vstack([x['loss'] for x in outputs])

        self._logfn(logits, preds, labels, mode="train epoch")

    def validation_epoch_end(self, outputs):

        logits = torch.vstack([x['logits'] for x in outputs])
        preds = torch.vstack([x['preds'] for x in outputs])
        labels = torch.vstack([x['labels'] for x in outputs])
        loss = torch.vstack([x['loss'] for x in outputs])

        self._logfn(logits, preds, labels, mode="val epoch")

    def test_epoch_end(self, outputs):

        logits = torch.vstack([x['logits'] for x in outputs])
        preds = torch.vstack([x['preds'] for x in outputs])
        labels = torch.vstack([x['labels'] for x in outputs])
        loss = torch.vstack([x['loss'] for x in outputs])

        self._logfn(logits, preds, labels, mode="test epoch")

    def predict_step(self, batch, batch_idx):

        imgs, labels, idxs = batch
        logits, preds = self.forward(imgs)

        return (preds, labels)

class NewRes(NeuralModel):

    def __init__(self, model_kwargs, lr, thresholds=8 * [0.5], pretrained=False, cp_path=None):

        super().__init__(model_kwargs, lr, thresholds)
        if pretrained:
            config = ResNetConfig.from_pretrained("microsoft/resnet-50")
        else:
            config = ResNetConfig(image_size=(224, 448), downsample_in_first_stage=False)

        self.model = ResNetForImageClassification(config)
        self.model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Flatten(start_dim=1, end_dim=-1), nn.Linear(2048, 8))

        state_dict = torch.load(cp_path)['state_dict']
        for key in list(state_dict.keys()):
            if 'model.' in key:
                state_dict[key.replace('model.', '')] = state_dict[key]
                del state_dict[key]
        state_dict.pop("classifier.2.weight")
        state_dict.pop("classifier.2.bias")
        self.model.load_state_dict(state_dict=state_dict, strict=False)
    def stats(self):
        size = (224, 248)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return (mean, std, size)

class ConvNext(NeuralModel):
    def __init__(self, model_kwargs, lr, thresholds= 8 * [0.5], pretrained=False, cp_path=None):

        super().__init__(model_kwargs, lr, thresholds)

        self.model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224",
                                                                ignore_mismatched_sizes=True,
                                                                label2id=label2id,
                                                                id2label=id2label)
        state_dict = torch.load(cp_path)['state_dict']
        for key in list(state_dict.keys()):
            if 'model.' in key:
                state_dict[key.replace('model.', '')] = state_dict[key]
                del state_dict[key]
        state_dict.pop("classifier.weight")
        state_dict.pop("classifier.bias")
        self.model.load_state_dict(state_dict=state_dict, strict=False)

    def stats(self):
        p = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
        mean, std, size = p.image_mean, p.image_std, (p.size['shortest_edge'], p.size['shortest_edge'])
        return (mean, std, size)

class BeIT(NeuralModel):

    def __init__(self, model_kwargs, lr, thresholds=8 * [0.5], pretrained = True, cp_path=None):

        super().__init__(model_kwargs, lr, thresholds)

        self.model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224",
                                                                    ignore_mismatched_sizes=True,
                                                                    label2id=label2id,
                                                                    id2label=id2label)

        state_dict = torch.load(cp_path)['state_dict']
        for key in list(state_dict.keys()):
            if 'model.' in key:
                state_dict[key.replace('model.', '')] = state_dict[key]
                del state_dict[key]
        state_dict.pop("classifier.weight")
        state_dict.pop("classifier.bias")
        self.model.load_state_dict(state_dict=state_dict, strict=False)

    def stats(self):
        # Using original mean and std
        p = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
        mean, std, size = p.image_mean, p.image_std, (p.size['height'], p.size['width'])
        return (mean, std, size)



EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-5

paths = [
    'load_checkpoints/beit_acc_checkpoint_pretrained/epoch=8-step=48942.ckpt',
    'load_checkpoints/beit_val_checkpoint_pretrained/epoch=4-step=27190.ckpt',
    'load_checkpoints/next_acc_checkpoint_pretrained/epoch=93-step=511172.ckpt',
    'load_checkpoints/next_val_checkpoint_pretrained/epoch=7-step=43504.ckpt',
    'load_checkpoints/resnet_acc_checkpoint/epoch=47-step=261024.ckpt',
    'load_checkpoints/resnet_val_checkpoint/epoch=5-step=32628.ckpt',
    'load_checkpoints/resnet_acc_checkpoint_untrained/epoch=87-step=478544.ckpt',
    'load_checkpoints/resnet_val_checkpoint_untrained/epoch=32-step=179454.ckpt',
]
names = [
    'beit_from_acc',
    'beit_from_val',
    'next_from_acc',
    'next_from_val',
    'resnet_from_acc',
    'resnet_from_val',
    'resnet_from_acc_untrained',
    'resnet_from_val_untrained',
]
is_pretrained = [True, True, True, True, True, True, False, False]
m = [BeIT, BeIT, ConvNext, ConvNext, NewRes, NewRes, NewRes, NewRes]
# ResNet
dataModule_resnet = LightningWrapper(annotation_file = ANNOTATION_FILE, batch_size = BATCH_SIZE, size = (224, 248)).prepare_data()
dataModule_beit = LightningWrapper(annotation_file = ANNOTATION_FILE, batch_size = BATCH_SIZE, size = (224, 248)).prepare_data()


cp_path = "load_checkpoints/resnet_acc_checkpoint/epoch=47-step=261024.ckpt"
res = NewRes(None, LEARNING_RATE, thresholds = 8*[0.5], pretrained= True, cp_path= cp_path)

#
SAVE_DIR = 'lightning_logs/'
checkpoint_callback = ModelCheckpoint(dirpath='checkpoint_path/', monitor="val_loss")

for model_class, cp_path, pretrained, name in zip(m, paths, is_pretrained, names):
    model = model_class(None, LEARNING_RATE, thresholds = 8*[0.5], pretrained= pretrained, cp_path= cp_path)
    mean, std, size = model.stats()
    dataModule = LightningWrapper(annotation_file=ANNOTATION_FILE, batch_size=BATCH_SIZE, mean = mean, std = std, size = size)
    dataModule.prepare_data()
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint_path/', name), monitor="val step loss")
    logger = CSVLogger("logs", version=name)

    print(name)

    trainer = Trainer(

        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        precision=16,

        max_epochs=EPOCHS,

        fast_dev_run=False,
        overfit_batches=False,
        #auto_lr_find=True,
        auto_scale_batch_size=False,  # set to 16 not to repeat process

        track_grad_norm=2,
        detect_anomaly=False,

        callbacks=[checkpoint_callback, LitProgressBar()],

        log_every_n_steps=50,

        gradient_clip_val=2,
        gradient_clip_algorithm="norm",

        val_check_interval=0.5,
        num_sanity_val_steps=-1,  # evaluate first
        logger=logger,
        )
    trainer.fit(model, dataModule)
#

pth = "checkpoint_path/beit_from_acc/epoch=11-step=10936.ckpt"
trainer.validate(model, dataModule, ckpt_path = pth)
results_test = trainer.predict(model, dataModule.test_dataloader(), ckpt_path = pth)
results_val = trainer.predict(model, dataModule.val_dataloader(), ckpt_path = pth)
mlss = MultilabelStatScores(num_labels=8, average = None)

y_hat, y = zip(*results_val)
y_hat, y = torch.vstack(y_hat), torch.vstack(y)
stats = mlss(y_hat, y)
for c, r in zip(CLASSES, stats):
    print(r, c)


with open('train_test_split_lib/stats.pkl', 'wb') as handle:
    pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


