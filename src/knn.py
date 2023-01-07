import os
import pickle

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from config import Config as CONFIG


def run_train_knn():

    all_embeddings = np.load(os.path.join(CONFIG.EXP_PATH, "all_embeddings.npy"))
    all_labels = np.load(os.path.join(CONFIG.EXP_PATH, "all_labels.npy"))

    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        algorithm="brute",
        leaf_size=30,
        metric="cosine",
    )
    knn.fit(all_embeddings, all_labels)
    with open(os.path.join(CONFIG.EXP_PATH, f"knn_{CONFIG.MODEL_NAME}.pkl"), "wb") as f:
        pickle.dump(knn, f)


if __name__ == "__main__":
    run_train_knn()
