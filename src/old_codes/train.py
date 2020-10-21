import os
from numpy.lib.shape_base import split
from sklearn import metrics
from sklearn import preprocessing, model_selection
import glob
import torch
import pandas as pd
import numpy as np

import config
import dataset
import engine
from models import CaptchaModel


def split(x):
    return [i for i in str(x)]


def run_training():
    # image_files
    image_files = glob.glob("../input/raw_captcha/*.png")
    print(image_files[:4])

    # targets
    targets_orig = [i.split("/")[-1][:-4] for i in image_files]
    print(targets_orig[:5])

    # creating a list of list for the targets
    targets = [[j for j in i] for i in targets_orig]

    # flattening the lists
    targets_flat = [item for sublists in targets for item in sublists]
    # print(targets_flat)

    lbl_encoder = preprocessing.LabelEncoder()
    lbl_encoder.fit(targets_flat)
    enc_targets = [lbl_encoder.transform(x) for x in targets]

    # this +1 is to add 1 to all the encoded labels, so that we could use 0 for the unknown values
    enc_targets = np.array(enc_targets) + 1
    print(len(enc_targets))
    print(len(lbl_encoder.classes_))

    (
        train_imgs,
        test_imgs,
        train_targets_orig,
        test_target_orig,
        train_targets,
        test_targets,
    ) = model_selection.train_test_split(
        image_files, targets_orig, enc_targets, test_size=0.1, random_state=42
    )

    print(len(train_imgs), len(train_targets))
    print(len(test_imgs), len(test_targets))
    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )

    test_dataset = dataset.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    # for data in train_dataloader:
    #     print(data)

    model = CaptchaModel(num_chars=len(lbl_encoder.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    for epoch in range(config.EPOCHS):
        train_loss = engine.train(model, train_dataloader, optimizer)
        valid_preds, valid_loss = engine.eval(model, test_dataloader)
        print(
            f"Epoch: {epoch}, train_loss: {train_loss}, val_loss: {valid_loss}")


if __name__ == "__main__":
    run_training()
