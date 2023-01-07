import copy
import os

import numpy as np
import pandas as pd
import torch
import torchvision
import wandb
from pytorch_metric_learning import losses
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from config import Config as CONFIG
from dataset import WhaleDataset
from utils import create_augmentations, evaluate_cos


def get_dataloaders():
    train_split = pd.read_csv("data/train_split.csv")
    val_split = pd.read_csv("data/val_split.csv")

    augmentations = create_augmentations(mode="train")

    train_dataset = WhaleDataset(train_split, 5005, augmentations)
    val_dataset = WhaleDataset(val_split, 5005, augmentations)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader

def train_model(vit=False):
    best_model = None
    best_recall = 0
    best_epoch = 0
    
    checkpoint_save_path = f"{CONFIG.EXP_PATH}/checkpoints"
    os.makedirs(checkpoint_save_path,exist_ok=True)
    
    for epoch in range(CONFIG.EPOCHS):

        model.train()
        train_loss = 0
        A = [[] for i in range(len(train_dataloader.dataset[0]))]
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch: {epoch}'):
            if vit:
                x_train, y_train = batch
                for i in x_train:
                    x_train[i] = x_train[i][:, 0].to(CONFIG.DEVICE)
                y_train = y_train.to(CONFIG.DEVICE)
            else:
                batch = tuple(t.to(CONFIG.DEVICE) for t in batch)
                x_train, y_train = batch

            optimizer.zero_grad()
            
            if vit:
                embeddings = model(**x_train).logits
            else:
                embeddings = model(x_train)

            loss = criterion(embeddings, y_train)
            loss.backward()
            optimizer.step()
            loss_optimizer.step()
            scheduler.step(epoch + step / iters)
            train_loss += loss.item()
        train_loss = train_loss / len(train_dataloader)
        print("Train Loss: {0:.5f}".format(train_loss))

        model.eval()

        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):          
                if vit:
                    x_val, y_val = batch
                    for i in x_val:
                        x_val[i] = x_val[i][:, 0].to(CONFIG.DEVICE)
                    y_val = y_val.to(CONFIG.DEVICE)
                else:
                    batch = tuple(t.to(CONFIG.DEVICE) for t in batch)
                    x_val, y_val = batch

                    if vit:
                        embeddings = model(**x_val).logits
                    else:
                        embeddings = model(x_val)

                loss = criterion(embeddings, y_val)
                val_loss += loss.item()
        val_loss = val_loss / len(val_dataloader)
        print("Loss Valid: {0:.5f}".format(val_loss))

        recalls = evaluate_cos(model, val_dataloader, CONFIG.DEVICE)
        if recalls[0] > best_recall:
            best_model = copy.deepcopy(model)
            best_recall = recalls[0]
            best_epoch = epoch
            torch.save(best_model, f'{checkpoint_save_path}/checkpoint_{best_recall}_{best_epoch}ep.pth')
        
        wandb.log(
            {
                f"train/Loss": np.float64(train_loss),
                f"val/Loss": np.float64(val_loss),
                "val/R@1": recalls[0],
                "val/R@2": recalls[1],
                "val/R@4": recalls[2],
                "val/R@8": recalls[3],
                "val/R@16": recalls[4],
                "val/R@32": recalls[5],
            }
        )

if __name__ == "__main__":

    wandb.init(
        project="whale_classification",
        name=f"{CONFIG.MODEL_NAME}_{CONFIG.LOSS_NAME}",
        config={
            "model": CONFIG.MODEL_NAME,
            "epochs": CONFIG.EPOCHS,
            "batch_size": CONFIG.BATCH_SIZE,
        },
    )
    os.makedirs(CONFIG.EXP_PATH, exist_ok=True)

    train_dataloader, val_dataloader = get_dataloaders()

    model = None
    if CONFIG.MODEL_NAME == 'ViT':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model.classifier = nn.Linear(768, CONFIG.EMBEDDING_SIZE)
        model.to(CONFIG.DEVICE)
    else:
        model = torchvision.models.resnet18(weights=None, progress=True)
        in_features = model.fc.in_features # classifier
        model.fc = nn.Linear(in_features, CONFIG.EMBEDDING_SIZE)
        model.to(CONFIG.DEVICE)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]

    criterion = losses.ProxyAnchorLoss(CONFIG.NUM_CLASSES, CONFIG.EMBEDDING_SIZE, margin = 0.1, alpha = 32).to(CONFIG.DEVICE)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=CONFIG.LR)
    iters = len(train_dataloader)
    scheduler = CosineAnnealingWarmRestarts(optimizer, iters)
    loss_optimizer = torch.optim.AdamW(criterion.parameters(), lr=0.00001)

    train_model()