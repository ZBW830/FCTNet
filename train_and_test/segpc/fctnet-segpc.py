
from __future__ import print_function, division

import os
import sys

sys.path.append('../..')
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

import copy
import json
import importlib
import glob
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.optim import Adam, SGD
from losses import DiceLoss, DiceLossWithLogtis
from torch.nn import BCELoss, CrossEntropyLoss

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")



# ## Set the seed
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
import random

random.seed(0)

# ## Load the config
CONFIG_FILE_PATH = '../configs/segpc/segpc2021_fctnet.yaml'

config = load_config(CONFIG_FILE_PATH)

print(json.dumps(config, indent=2))
print(20 * "~-", "\n")

# ## Dataset and Dataloader

from datasets.segpc import SegPC2021Dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

def load_config(config_filepath):
    try:
        with open(config_filepath, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)

# ----------------- transform ------------------
# transform for image
img_transform = transforms.Compose([
    transforms.ToTensor()
])
# transform for mask
msk_transform = transforms.Compose([
    transforms.ToTensor()
])
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ----------------- dataset --------------------
# preparing training dataset
tr_dataset = SegPC2021Dataset(
    mode="tr",
    input_size=config['dataset']['input_size'],
    scale=config['dataset']['scale'],
    dataset_dir=config['dataset']['dataset_dir'],
    one_hot=True,
    force_rebuild=False,
    img_transform=img_transform,
    msk_transform=msk_transform
)
# preparing training dataset
vl_dataset = SegPC2021Dataset(
    mode="vl",
    input_size=config['dataset']['input_size'],
    scale=config['dataset']['scale'],
    dataset_dir=config['dataset']['dataset_dir'],
    one_hot=True,
    force_rebuild=False,
    img_transform=img_transform,
    msk_transform=msk_transform
)
# preparing training dataset
te_dataset = SegPC2021Dataset(
    mode="te",
    input_size=config['dataset']['input_size'],
    scale=config['dataset']['scale'],
    dataset_dir=config['dataset']['dataset_dir'],
    one_hot=True,
    force_rebuild=False,
    img_transform=img_transform,
    msk_transform=msk_transform
)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

# prepare train dataloader
tr_dataloader = DataLoader(tr_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['train'])

# prepare validation dataloader
vl_dataloader = DataLoader(vl_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['validation'])

# prepare test dataloader
te_dataloader = DataLoader(te_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['test'])

print(f"Length of trainig_dataset:\t{len(tr_dataset)}")
print(f"Length of validation_dataset:\t{len(vl_dataset)}")
print(f"Length of test_dataset:\t\t{len(te_dataset)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch device: {device}")


metrics = torchmetrics.MetricCollection(
    [
        torchmetrics.F1Score(task='binary'),
        torchmetrics.Accuracy(task='binary'),
        torchmetrics.Dice(),
        torchmetrics.Precision(task='binary'),
        torchmetrics.Recall(task='binary'),
        torchmetrics.Specificity(task='binary'),
        # IoU
        torchmetrics.JaccardIndex(task='multiclass', num_classes=2)
    ],
    prefix='train_metrics/'
)

# train_metrics
train_metrics = metrics.clone(prefix='train_metrics/').to(device)

# valid_metrics
valid_metrics = metrics.clone(prefix='valid_metrics/').to(device)

# test_metrics
test_metrics = metrics.clone(prefix='test_metrics/').to(device)



def make_serializeable_metrics(computed_metrics):
    res = {}
    for k, v in computed_metrics.items():
        res[k] = float(v.cpu().detach().numpy())
    return res


# ## Define validate function


def validate(model, criterion, vl_dataloader):
    model.eval()
    with torch.no_grad():
        evaluator = valid_metrics.clone().to(device)

        losses = []
        cnt = 0.
        for batch, batch_data in enumerate(vl_dataloader):
            imgs = batch_data['image']
            msks = batch_data['mask']

            cnt += msks.shape[0]

            imgs = imgs.to(device)
            msks = msks.to(device)

            preds = model(imgs)
            loss = criterion(preds, msks)
            losses.append(loss.item())

            # we should remove nucs
            not_nucs = torch.where(imgs[:, -1, :, :] > 0, 0, 1)
            preds_en = torch.argmax(preds, 1, keepdim=False).float() * not_nucs
            msks_en = torch.argmax(msks, 1, keepdim=False) * not_nucs

            # evaluate by metrics
            evaluator.update(preds_en, msks_en.int())

        loss = np.sum(losses) / cnt
        metrics = evaluator.compute()

    return evaluator, loss


# ## Define train function

# In[14]:


def train(
        model,
        device,
        tr_dataloader,
        vl_dataloader,
        config,

        criterion,
        optimizer,
        scheduler,

        save_dir='./',
        save_file_id=None,
):
    tr_prms = config['training']
    EPOCHS = tr_prms['epochs']

    torch.cuda.empty_cache()
    model = model.to(device)

    evaluator = train_metrics.clone().to(device)

    epochs_info = []
    best_model = None
    best_result = {}
    best_vl_loss = np.Inf
    for epoch in range(EPOCHS):
        model.train()

        evaluator.reset()
        tr_iterator = tqdm(enumerate(tr_dataloader))
        tr_losses = []
        cnt = 0
        for batch, batch_data in tr_iterator:
            imgs = batch_data['image']
            msks = batch_data['mask']

            imgs = imgs.to(device)
            msks = msks.to(device)

            optimizer.zero_grad()
            preds= model(imgs)

            loss = criterion(preds, msks)
            loss.backward()
            optimizer.step()

            # we should remove nucs
            not_nucs = torch.where(imgs[:, -1, :, :] > 0, 0, 1)
            preds_en = torch.argmax(preds, 1, keepdim=False).float() * not_nucs
            msks_en = torch.argmax(msks, 1, keepdim=False) * not_nucs

            # evaluate by metrics
            evaluator.update(preds_en, msks_en.int())

            cnt += imgs.shape[0]
            tr_losses.append(loss.item())

            # write details for each training batch
            _cml = f"curr_mean-loss:{np.sum(tr_losses) / cnt:0.5f}"
            _bl = f"mean_batch-loss:{tr_losses[-1] / imgs.shape[0]:0.5f}"
            tr_iterator.set_description(f"Training) ep:{epoch:03d}, batch:{batch + 1:04d} -> {_cml}, {_bl}")

        tr_loss = np.sum(tr_losses) / cnt

        # validate model
        vl_metrics, vl_loss = validate(model, criterion, vl_dataloader)
        if vl_loss < best_vl_loss:
            # find a better model
            best_model = model
            best_vl_loss = vl_loss
            best_result = {
                'tr_loss': tr_loss,
                'vl_loss': vl_loss,
                'tr_metrics': make_serializeable_metrics(evaluator.compute()),
                'vl_metrics': make_serializeable_metrics(vl_metrics.compute())
            }

        # write the final results
        epoch_info = {
            'tr_loss': tr_loss,
            'vl_loss': vl_loss,
            'tr_metrics': make_serializeable_metrics(evaluator.compute()),
            'vl_metrics': make_serializeable_metrics(vl_metrics.compute())
        }
        epochs_info.append(epoch_info)
        #         epoch_tqdm.set_description(f"Epoch:{epoch+1}/{EPOCHS} -> tr_loss:{tr_loss}, vl_loss:{vl_loss}")
        evaluator.reset()

        scheduler.step(vl_loss)

    # save final results
    res = {
        'id': save_file_id,
        'config': config,
        'epochs_info': epochs_info,
        'best_result': best_result
    }
    fn = f"{save_file_id + '_' if save_file_id else ''}result.json"
    fp = os.path.join(config['model']['save_dir'], fn)
    with open(fp, "w") as write_file:
        json.dump(res, write_file, indent=4)

    # save model's state_dict
    fn = "last_model_state_dict.pt"
    fp = os.path.join(config['model']['save_dir'], fn)
    torch.save(model.state_dict(), fp)

    # save the best model's state_dict
    fn = "best_model_state_dict.pt"
    fp = os.path.join(config['model']['save_dir'], fn)
    torch.save(best_model.state_dict(), fp)

    return best_model, model, res


# ## Define test function

def test(model, te_dataloader):
    model.eval()
    with torch.no_grad():
        evaluator = test_metrics.clone().to(device)
        for batch_data in tqdm(te_dataloader):
            imgs = batch_data['image']
            msks = batch_data['mask']

            imgs = imgs.to(device)
            msks = msks.to(device)

            preds = model(imgs)
            # preds = model(imgs)

            # we should remove nucs
            not_nucs = torch.where(imgs[:, -1, :, :] > 0, 0, 1)
            preds_en = torch.argmax(preds, 1, keepdim=False).float() * not_nucs
            msks_en = torch.argmax(msks, 1, keepdim=False) * not_nucs

            # evaluate by metrics
            evaluator.update(preds_en, msks_en.int())
    return evaluator


## Load and prepare model

# In[17]:

from models.FCT_Net import FCTNet
model = FCTNet(in_chans=4, num_classes=2, gt_ds=False, sigmoid=True)
torch.cuda.empty_cache()
model = model.to(device)
model = nn.DataParallel(model)
print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

os.makedirs(config['model']['save_dir'], exist_ok=True)
model_path = f"{config['model']['save_dir']}/model_state_dict.pt"

if config['model']['load_weights']:
    model.load_state_dict(torch.load(model_path))
    print("Loaded pre-trained weights...")


criterion_dice = DiceLossWithLogtis()

criterion_ce = CrossEntropyLoss()


def criterion(preds, masks):
    c_dice = criterion_dice(preds, masks)
    c_ce = criterion_ce(preds, masks)
    return 0.5 * c_dice + 0.5 * c_ce


tr_prms = config['training']
optimizer = globals()[tr_prms['optimizer']['name']](model.parameters(), **tr_prms['optimizer']['params'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **tr_prms['scheduler'])

# ## Start traning

best_model, model, res = train(
    model,
    device,
    tr_dataloader,
    vl_dataloader,
    config,

    criterion,
    optimizer,
    scheduler,

    save_dir=config['model']['save_dir'],
    save_file_id=None,
)


te_metrics = test(best_model, te_dataloader)
te_metrics.compute()


f"{config['model']['save_dir']}"

# Test the best inferred model


## Load the best model

best_model = FCTNet(in_chans=4, num_classes=2, gt_ds=False, sigmoid=True)
torch.cuda.empty_cache()
best_model = best_model.to(device)
best_model = nn.DataParallel(best_model)


fn = "best_model_state_dict.pt"
os.makedirs(config['model']['save_dir'], exist_ok=True)
model_path = f"{config['model']['save_dir']}/{fn}"

best_model.load_state_dict(torch.load(model_path))
print("Loaded best model weights...")

# ## Evaluation

te_metrics = test(best_model, te_dataloader)
print(te_metrics.compute())
# ## Plot graphs

result_file_path = f"{config['model']['save_dir']}/result.json"
with open(result_file_path, 'r') as f:
    results = json.loads(''.join(f.readlines()))
epochs_info = results['epochs_info']


# ## Save images

from PIL import Image

save_imgs_dir = f"{config['model']['save_dir']}/visualized"

if not os.path.isdir(save_imgs_dir):
    os.mkdir(save_imgs_dir)

with torch.no_grad():
    for batch in tqdm(te_dataloader):
        imgs = batch['image']
        msks = batch['mask']
        ids = batch['id']

        preds = best_model(imgs.to(device))

        txm = imgs.cpu().numpy()
        tbm = torch.argmax(msks, 1).cpu().numpy()
        tpm = torch.argmax(preds, 1).cpu().numpy()
        tid = ids.cpu().numpy()

        for idx in range(len(tbm)):
            img = np.moveaxis(txm[idx, :3], 0, -1) * 255.
            nuc = np.uint8(txm[idx, -1] * 255.)
            gt = np.uint8(tbm[idx] * 255.)
            gt_3ch = np.stack([gt - nuc, nuc * 0, nuc], -1)
            pred = np.where(tpm[idx] > 0.5, 255, 0)
            pred_3ch = np.stack([pred - nuc, nuc * 0, nuc], -1)

            img = np.ascontiguousarray(img, dtype=np.uint8)
            gt_3ch = np.ascontiguousarray(gt_3ch, dtype=np.uint8)
            pred_3ch = np.ascontiguousarray(pred_3ch, dtype=np.uint8)

            fid = tid[idx]
            Image.fromarray(img).save(f"{save_imgs_dir}/{fid}_img.png")
            Image.fromarray(gt_3ch).save(f"{save_imgs_dir}/{fid}_gt.png")
            Image.fromarray(pred_3ch).save(f"{save_imgs_dir}/{fid}_pred.png")

