import pytorch_lightning as pl
import os
from numpy.lib.shape_base import split
from sklearn import metrics
from sklearn import preprocessing, model_selection
import glob
import torch
import pandas as pd
import io
import numpy as np
import torch.nn as nn
import joblib
from PIL import Image


class CaptchaModel(pl.LightningModule):
    def __init__(self, num_classes):
        super(CaptchaModel, self).__init__()

        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear_1 = nn.Linear(1024, 64)
        self.drop_1 = nn.Dropout(0.2)

        self.gru = nn.GRU(
            64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True
        )
        self.output = nn.Linear(64, num_classes + 1)

    def forward(self, images, targets=None):
        bs, ch, ht, wd = images.size()
        # print(bs, ch, ht, wd)
        x = F.relu(self.conv_1(images))
        # print(x.size())
        x = self.max_pool1(x)
        # print(x.size())

        x = F.relu(self.conv_2(x))
        # print(x.size())
        x = self.max_pool2(x)  # 1, 64, 18, 75
        # print(x.size())  # before passing these outputs into custom rnn permute the outputs (0, 3, 1, 2)
        x = x.permute(
            0, 3, 1, 2
        )  # 1, 75, 64, 18   # because we have to go through the width of the images
        # print("1st permute: ", x.size())
        x = x.view(bs, x.size(1), -1)
        # print(x.size())
        x = self.linear_1(x)
        x = self.drop_1(x)
        # print(x.size())
        x, _ = self.gru(x)
        # print(x.size())
        x = self.output(x)
        # print(x.size())
        # To calculate the ctc loss, we should again permute it
        # this you have to remember, timestamps, batches, values
        x = x.permute(1, 0, 2)
        # print(x.shape)

        if targets is not None:
            # ctc loss is already implemented in pytorch, but it is not straight forward.
            # it takes log softmax values.
            log_softmax_values = F.log_softmax(
                x, 2
            )  # (x, 2) indicates, x th second index which is num_classes + 1

            # Two things have to specified here, length of inputs and len of outputs
            input_lengths = torch.full(
                size=(bs,), fill_value=log_softmax_values.size(0), dtype=torch.int32
            )
            # print(input_lengths)
            targets_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            # print(targets_lengths)
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, targets, input_lengths, targets_lengths
            )

            return x, loss

        return x, None


lbl_enc = joblib.load(
    "../input/pickles/tan_pan_oltas_gst_epfo_lbl_encoder.pkl")
model = CaptchaModel(num_classes=len(lbl_enc.classes_))
model = model.load_from_checkpoint(
    "../src/lightning_logs/version_0/checkpoints/epoch=105.ckpt", num_classes=len(lbl_enc.classes_))


trainer = pl.Trainer(gpus=1, max_epochs=200)
trainer.test(model, )
