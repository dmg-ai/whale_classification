import os
import pickle

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import Config as CONFIG


def prepare_data():
    train = pd.read_csv("data/train.csv")
    train.columns = ["image", "id"]

    # images count of each class
    vals_counts = (
        train["id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="counts")
    )

    # class where images count equal 1
    class_eq1 = vals_counts[vals_counts["counts"] == 1]["unique_values"].values  # class_eq1.shape == (2073, 2)

    # label encoder ;)
    le = {}
    classes = train["id"].unique()
    for i in range(classes.shape[0]):
        le[classes[i]] = i

    with open(os.path.join(CONFIG.EXP_PATH,'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
        
    train["target"] = train["id"].map(le)

    return train, class_eq1


def create_splits():
    train, class_eq1 = prepare_data()

    train_val_part = train[~train["id"].isin(class_eq1) & (train["id"] != "new_whale")]  # not new_whale and img count by class more than 1
    train_part = train[train["id"].isin(class_eq1)]  # img count by class equal 1
    only_new_whale = train[train["id"] == "new_whale"]  # only new_whale class

    # make folds to have equal number of images in train and val splits by class
    # among not "new_whale" class and images count by class more than 1
    skf = StratifiedKFold(n_splits=2)
    train_val_part = train_val_part.reset_index().drop("index", axis=1)

    for fold, (train_index, val_index) in enumerate(skf.split(train_val_part.drop("target", axis=1), train_val_part["target"])):
        train_val_part.loc[val_index, "fold"] = fold
    train_val_part["fold"] = train_val_part["fold"].astype(int)

    # split "new_whale" class into train and val
    new_whale_train, new_whale_val = train_test_split(only_new_whale, test_size=0.2)

    # combine all data for train split
    train_split = pd.concat([train_val_part[train_val_part["fold"] == 0], train_part, new_whale_train],axis=0)
    train_split = train_split.reset_index().drop("index", axis=1)  # len(train_split) == 16616

    # combine all data for val split
    val_split = pd.concat([train_val_part[train_val_part["fold"] == 1], new_whale_val], axis=0)
    val_split = val_split.reset_index().drop("index", axis=1)  # len(val_split) == 8745

    train_split.to_csv("data/train_split.csv", index=False)
    val_split.to_csv("data/val_split.csv", index=False)

    print('Successfully! Splits were saved into "data" folder.')


if __name__ == "__main__":
    create_splits()
