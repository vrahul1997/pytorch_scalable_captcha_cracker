import glob
import os

import numpy as np
import pytorch_lightning as pl
import torch
import joblib
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.metrics.functional import accuracy
from sklearn import model_selection, preprocessing

import config
import dataset
from models import CaptchaModel

single_test_file = glob.glob("input/single_test/*.png")

model_chkpoint = "src/lightning_logs/version_3/checkpoints/epoch=86.ckpt"
lbl_encoder = joblib.load(
    "input/pickles/tan_pan_oltas_gst_epfo_rc_lbl_encoder.pkl")


def remove_duplicates(x):
    letter = None
    word = []
    for i in x:
        if i == "$":
            letter = None
        if i != "$" and i != letter:
            letter = None
        if i != "$" and letter is None:
            letter = i
            word.append(i)
        if i != "$" and letter is not None:
            pass
    word = "".join(word)
    return word


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


class CaptchaModel(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 train_dataset=None,
                 val_dataset=None,
                 test_dataset=None,
                 test_targets_orig=None,
                 lbl_encoder=None):
        super(CaptchaModel, self).__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.lbl_encoder = lbl_encoder
        self.test_targets_orig = test_targets_orig

        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear_1 = nn.Linear(1024, 64)
        self.drop_1 = nn.Dropout(0.2)

        self.gru = nn.GRU(64,
                          32,
                          bidirectional=True,
                          num_layers=2,
                          dropout=0.25,
                          batch_first=True)
        self.output = nn.Linear(64, num_classes + 1)

    def forward(self, images, targets=None):
        bs, ch, ht, wd = images.size()
        x = F.relu(self.conv_1(images))
        x = self.max_pool1(x)

        x = F.relu(self.conv_2(x))
        x = self.max_pool2(x)  # 1, 64, 18, 75
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = self.linear_1(x)
        x = self.drop_1(x)
        x, _ = self.gru(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)

        if targets is not None:
            log_softmax_values = F.log_softmax(x, 2)
            input_lengths = torch.full(size=(bs, ),
                                       fill_value=log_softmax_values.size(0),
                                       dtype=torch.int32)
            targets_lengths = torch.full(size=(bs, ),
                                         fill_value=targets.size(1),
                                         dtype=torch.int32)
            loss = nn.CTCLoss(blank=0)(log_softmax_values, targets,
                                       input_lengths, targets_lengths)
            return x, loss
        return x, None

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = torch.optim.Adam(params, lr=3e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, targets = batch["images"], batch["targets"]
        logits, loss = self(images, targets)
        return {"loss": loss}

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )
        return val_loader

    def validation_step(self, batch, batch_idx):
        images, targets = batch["images"], batch["targets"]
        logits, loss = self(images, targets)
        return {"loss": loss, "logits": logits}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x["loss"]
                                     for x in val_step_outputs]).mean()
        logits = [x["logits"] for x in val_step_outputs]
        valid_captcha_preds = []
        for vp in logits:
            current_preds = decode_predictions(vp, self.lbl_encoder)
            valid_captcha_preds.extend(current_preds)
        combined = list(zip(self.test_targets_orig, valid_captcha_preds))
        print(combined[:20])
        pbar = {"val_loss": avg_val_loss}
        return {"progress_bar": pbar}

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )
        return test_loader

    def test_step(self, batch, batch_idx):
        images = batch["images"]
        logits, loss = self(images)
        return {"loss": loss, "logits": logits}

    def test_epoch_end(self, test_step_outputs):
        logits = [x["logits"] for x in test_step_outputs]
        test_captcha_preds = []
        for vp in logits:
            current_preds = decode_predictions(vp, self.lbl_encoder)
            test_captcha_preds.extend(current_preds)
        self.fin_res = test_captcha_preds
        return {"logits": self.fin_res}

    def return_final_output(self):
        return self.fin_res


def pred_captcha(img_bytes):
    image = Image.open(img_bytes).convert("RGB")  #change accordingly
    test_dataset = dataset.ClassificationDataset(
        image_paths=image,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    trainer = pl.Trainer(max_epochs=200)
    model = CaptchaModel(
        num_classes=len(lbl_encoder.classes_),
        test_dataset=test_dataset,
        lbl_encoder=lbl_encoder,
    )
    model.load_state_dict(
        torch.load(model_chkpoint,
                   map_location=torch.device('cpu'))['state_dict'])
    trainer.test(model)
    fin_res = model.return_final_output()
    print(fin_res)
