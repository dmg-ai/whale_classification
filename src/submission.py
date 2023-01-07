import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import ViTFeatureExtractor

from config import Config as CONFIG
from utils import create_augmentations, predict_embedding, read_image

def create_sub():
    model = torch.load(CONFIG.MODEL_CHECKPOINT, map_location=CONFIG.DEVICE)

    with open(os.path.join(CONFIG.EXP_PATH, f"knn_{CONFIG.MODEL_NAME}.pkl"), 'rb') as f:
        knn = pickle.load(f)

    with open(os.path.join(CONFIG.EXP_PATH,'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
        
    all_labels = np.load(os.path.join(CONFIG.EXP_PATH, "all_labels.npy"))

    augmentations = create_augmentations()

    feature_extractor = None
    vit_flag = False
    if CONFIG.MODEL_NAME == 'ViT':
        vit_flag = True
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    test_images = [i for i in os.listdir('data/test') if i.endswith('.jpg')]
    result_dict = {}
    for img in tqdm(test_images):
        image_filepath = os.path.join('data/test', img)
        image = read_image(image_filepath, augmentations, CONFIG.DEVICE, feature_extractor=feature_extractor)
        
        embedding = predict_embedding(model, image, vit=vit_flag)
        
        neighs = knn.kneighbors([embedding],5)
        knn_nearest_5_classes = all_labels[neighs[1][0]]
        
        nearest_labels = []
        for target in knn_nearest_5_classes:
            nearest_labels.append(list(le.keys())[list(le.values()).index(target)])
        result_dict[img] = ' '.join(list(dict.fromkeys(nearest_labels)))

    sub = pd.DataFrame.from_dict(result_dict, orient='index', columns=['Id'])
    sub = sub.reset_index()
    sub.columns = ['Image', 'Id']

    sub.to_csv(os.path.join(CONFIG.EXP_PATH,f'{CONFIG.MODEL_NAME}/knn_submission.csv'), index=False)

if __name__ == "__name__":
    create_sub()