import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


def read_image(image, augmentations=None, device='cpu', feature_extractor=None):
    if type(image) is str:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else: # if image is PIL
        image = np.array(image.convert('RGB')) 

    if augmentations:
        image = augmentations(image=image)['image']
    else:
        image = torch.Tensor(image)

    if feature_extractor:
        image = feature_extractor(images=image, return_tensors="pt")
    else:
        image = image.unsqueeze(0)
        image = image.float()
    image = image.to(device)
    return image

def predict_embedding(model, image, vit=False):
    if vit:
        embedding = model(**image).logits
    else:
        embedding = model(image)
    embedding = embedding.detach().cpu().numpy()
    embedding = np.squeeze(embedding, axis=0)
    return embedding

def create_augmentations(mode='train'):
    if mode =='train':
        return A.Compose(
            [
                A.Resize(512, 512),
                # A.CLAHE(p=1),
                ToTensorV2(),
            ]
        )
    
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))

def predict_batchwise(model, dataloader, device):
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.to(device))

                for j in J:
                    A[i].append(j)# A[0] содержит список из BATCH_SIZE тензоров-эмбеддингов
                                   # A[1] содержит список из BATCH_SIZE тензоров таргетов
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def predict_batchwise_vit(model, dataloader, device):
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                                
                if i == 0:
                    for k in J:
                        J[k] = J[k][:, 0].to(device)
                    J = model(**J).logits

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    return [torch.stack(A[i]) for i in range(len(A))]

def evaluate_cos(model, dataloader, device, vit=False):
    # calculate embeddings with model and get targets
    if vit:
        X, T = predict_batchwise_vit(model, dataloader, device)
    else:
        X, T = predict_batchwise(model, dataloader, device)

    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

