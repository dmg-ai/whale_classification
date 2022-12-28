import os
import pickle
import warnings

import gradio as gr
import torch
import numpy as np
import pandas as pd
from transformers import ViTFeatureExtractor

from src.utils import create_augmentations, predict_embedding, read_image

warnings.filterwarnings("ignore")

logo_URL = "https://itmo.ru/file/pages/213/logo_osnovnoy_russkiy_chernyy.jpg"
HEADLINE_IMAGE = "<center> <img src= {} width=200px></center>".format(logo_URL)
HEADLINE = "Whale Classification \U0001F929 \U0001F433" # 
RADIO_BTN_VERSIONS = ['resnet18 (🔺 speed, 🔻 robustness)','ViT (🔻 speed, 🔺 robustness)']
INFERENCE_EXAMPLES = [['examples/new_whale.jpg','new_whale'],['examples/w_f48451c.jpg','w_f48451c']]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESNET_MODEL = torch.load('models/resnet18/checkpoint_resnet18_proxy.pth', map_location=DEVICE)
with open('models/resnet18/knn_resnet18.pkl', 'rb') as f:
    RESNET_KNN = pickle.load(f)
RESNET_LABELS = np.load("models/resnet18/all_labels.npy")


VIT_MODEL = torch.load('models/vit/checkpoint_vit_proxy.pth', map_location=DEVICE)
with open('models/vit/knn_vit.pkl', 'rb') as f:
    VIT_KNN = pickle.load(f)
VIT_LABELS = np.load("models/vit/all_labels.npy")

with open(os.path.join('models/label_encoder.pkl'), 'rb') as f:
    LABEL_ENCODER = pickle.load(f)


def inference(image, model_name):
    if model_name == RADIO_BTN_VERSIONS[0]:
        model = RESNET_MODEL
        knn = RESNET_KNN
        all_labels = RESNET_LABELS
    else:
        model = VIT_MODEL
        knn = VIT_KNN
        all_labels = VIT_LABELS

    augmentations = create_augmentations()

    feature_extractor = None
    vit_flag = False
    if model_name == RADIO_BTN_VERSIONS[1]:
        vit_flag = True
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    image = read_image(image, augmentations, DEVICE, feature_extractor=feature_extractor)
    
    embedding = predict_embedding(model, image, vit=vit_flag)
    
    neighs_info = knn.kneighbors([embedding],5)
    distances = neighs_info[0][0]
    knn_nearest_5_classes = all_labels[neighs_info[1][0]]
    nearest_labels = []
    for target in knn_nearest_5_classes:
        nearest_labels.append(list(LABEL_ENCODER.keys())[list(LABEL_ENCODER.values()).index(target)])
    
    res_dict = dict()
    for target, dist in zip(nearest_labels, distances):
        res_dict[target] = float(dist)   
    sub = pd.DataFrame.from_dict(res_dict, orient='index', columns=['Id'])
    sub = sub.reset_index()
    sub.columns = ['Label', 'Cos Distance']
    return sub

demo = gr.Interface(
    fn=inference,
    inputs=[gr.inputs.Image(type="pil"),
            gr.inputs.Radio(RADIO_BTN_VERSIONS, 
            type="value", default=RADIO_BTN_VERSIONS[1], label='model')], 
    outputs=gr.Dataframe(type="pandas", headers=['Label', 'Cos Distance'], row_count=5, col_count=2),
    title=HEADLINE,
    description=HEADLINE_IMAGE,
    #article=article,
    examples=INFERENCE_EXAMPLES,
    allow_flagging='never')

demo.launch()