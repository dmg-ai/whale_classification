import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import ViTFeatureExtractor

from config import Config as CONFIG
from utils import create_augmentations, predict_embedding, read_image

def get_embeddings():
    train = pd.read_csv('data/train.csv')
    model = torch.load(CONFIG.MODEL_CHECKPOINT, map_location=CONFIG.DEVICE)

    augmentations = create_augmentations()

    all_embeddings = []
    all_labels = []

    feature_extractor = None
    vit_flag = False
    if CONFIG.MODEL_NAME == 'ViT':
        vit_flag = False
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    for i in tqdm(range(train.shape[0])):
        obj = train.loc[i]
        image_filepath = os.path.join('data/train', obj['image'])
        image = read_image(image_filepath, augmentations, CONFIG.DEVICE, feature_extractor=feature_extractor)
        
        label = obj['target']
        
        embedding = predict_embedding(model, image, vit=vit_flag)
        
        all_embeddings.append(embedding)
        all_labels.append(label)

        all_embeddings = np.array(all_embeddings)
        all_labels = np.array(all_labels)

        with open(os.path.join(CONFIG.EXP_PATH,'all_embeddings.npy'), 'wb') as f:
            np.save(f, all_embeddings)
        with open(os.path.join(CONFIG.EXP_PATH,'all_labels.npy'), 'wb') as f:
            np.save(f, all_labels)
