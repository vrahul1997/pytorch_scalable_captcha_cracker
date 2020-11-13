import os
from numpy.lib.shape_base import split
from sklearn import metrics
from sklearn import preprocessing, model_selection
import glob
import torch
import pandas as pd
import numpy as np
import joblib

import dataset
from models import CaptchaModel
import config


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    try:
        fin = fin.replace("#", "")
    except Exception as e:
        pass
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds


def run_test(u, v):
    image_files = sorted(
        glob.glob("../input/all_captcha_types/mca_captcha/test_images/*.png")
    )[u:v]
    print(image_files[:5])

    test_dataset = dataset.ClassificationDataset(
        image_paths=image_files,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )
    lbl_enc = joblib.load("../input/pickles/tan_pan_oltas_gst_epfo_rc_lbl_encoder.pkl")
    model = CaptchaModel(len(lbl_enc.classes_))

    model.load_state_dict(
        torch.load("../src/lightning_logs/version_3/checkpoints/epoch=86.ckpt")[
            "state_dict"
        ]
    )

    test_preds = []
    for data in test_loader:
        model.eval()
        batch_preds, _ = model(**data)
        test_preds.append(batch_preds)

    all_preds = []
    for test_data in test_preds:
        current_preds = decode_predictions(test_data, lbl_enc)
        for i in current_preds:
            all_preds.append(i)
    print(all_preds)
    return all_preds
    # df = pd.read_csv("../input/test_images_outputs.csv")
    # df['pytorch_preds'] = all_preds
    # df.to_csv("../input/comparison.csv")
    # print(len(df[df['Captcha Value'] != df['pytorch_preds']]['Captcha Value']))


if __name__ == "__main__":
    final_preds_mca = []
    frst_250 = run_test(0, 100)
    final_preds_mca.append(frst_250)
    # second_250 = run_test(250, 500)
    # final_preds_mca.append(second_250)
    final_preds_mca = [item for sublists in final_preds_mca for item in sublists]
    images = sorted(
        glob.glob("../input/all_captcha_types/mca_captcha/test_images/*.png")
    )
    inputs = [i.split("/")[-1].split(".")[0] for i in images]
    df = pd.DataFrame(data=zip(inputs, final_preds_mca), columns=["values", "preds"])
    df.to_csv("preds_mca.csv", index=False)
    print(df)
