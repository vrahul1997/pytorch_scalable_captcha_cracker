import os
from numpy.lib.shape_base import split
from sklearn import metrics
from sklearn import preprocessing, model_selection
import glob
import torch
import pandas as pd
import io
import numpy as np
import joblib
from PIL import Image

import dataset
from models import CaptchaModel
import config


class CaptchaPred:

    def __init__(self, img_bytes):
        self.image_bytes = img_bytes
        self.get_pred(self.image_bytes)

    def remove_duplicates(self, x):
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


    def decode_predictions(self, preds, encoder):
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
            cap_preds.append(self.remove_duplicates(tp))
        return cap_preds

    def get_pred(self, img_bytes):
        # img = Image.open("../input/mca_test_files/3487.png", mode='r')
        # imgByteArr = io.BytesIO()
        # img.save(imgByteArr, format='PNG')
        # imgByteArr = imgByteArr.getvalue()
        image = (Image.open(io.BytesIO(img_bytes)).convert("RGB"), )


        test_dataset = dataset.ClassificationDataset(
            image_paths=image,
            resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
        )
        lbl_enc = joblib.load("../input/pickles/lbl_encoder.pkl")
        model = CaptchaModel(36)
        model.load_state_dict(torch.load("../input/pickles/captcha.pth"))

        test_preds = []
        for data in test_loader:
            model.eval()
            batch_preds, _ = model(**data)
            test_preds.append(batch_preds)
        

        all_preds = []
        for test_data in test_preds:
            current_preds = self.decode_predictions(test_data, lbl_enc)
            for i in current_preds:
                all_preds.append(i)
        print(all_preds[0])
        # df = pd.read_csv("../input/test_images_outputs.csv")
        # df['pytorch_preds'] = all_preds
        # df.to_csv("../input/comparison.csv")
        # print(len(df[df['Captcha Value'] != df['pytorch_preds']]['Captcha Value']))


# img = Image.open("../input/mca_test_files/3489.png", mode='r')
# imgByteArr = io.BytesIO()
# img.save(imgByteArr, format='PNG')
# imgByteArr = imgByteArr.getvalue()

CaptchaPred(imgByteArr)