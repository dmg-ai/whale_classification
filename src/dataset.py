import os

import cv2
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor


class WhaleDataset(Dataset):
    def __init__(self, df, transform=None, vit=False):
        self.df = df
        self.transform = transform
        self.vit = vit
        if self.vit:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_filepath = os.path.join("data/train", self.df["image"][idx])
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df["target"][idx]

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.vit:
            image = self.feature_extractor(images=image, return_tensors="pt")
        else:
            image = image.float()

        return image, label
