import os

import torch

class Config:
    MODEL_NAME = "ViT"
    LOSS_NAME = "proxy"
    EPOCHS = 50
    BATCH_SIZE = 8
    LR = 2e-5
    NUM_CLASSES = 5005
    EMBEDDING_SIZE = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXP_PATH = f"experiments/{LOSS_NAME}/{MODEL_NAME}"
    MODEL_CHECKPOINT = os.path.join(EXP_PATH, "models/vit/checkpoint_vit_proxy.pth")
