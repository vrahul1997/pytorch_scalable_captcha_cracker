from models import CaptchaModel
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn import preprocessing, model_selection
import numpy as np
import glob
import dataset
import config


class CaptchaModel(pl.LightningModule):
    def __init__(self, num_chars=36):
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
        self.output = nn.Linear(64, num_chars + 1)

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
            )  # (x, 2) indicates, x th second index which is num_chars + 1

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

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = torch.optim.Adam(params, lr=3e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, targets = batch['images'], batch['targets']
        logits, loss = self(images, targets)
        return {"loss": loss}

    def train_dataloader(self):
        # image_files
        image_files = glob.glob("../input/captcha_images_v2/*.png")
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
        enc_targets = np.array(enc_targets)
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
            image_files, targets_orig, enc_targets, test_size=0.2, random_state=42
        )
        train_dataset = dataset.ClassificationDataset(
            image_paths=train_imgs,
            targets=train_targets,
            resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
        )

        test_dataset = dataset.ClassificationDataset(
            image_paths=test_imgs,
            targets=test_targets,
            resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
        )
        return self.train_dataloader

    def val_dataloader(self):
        dataloader = self.val_dataloader
        return dataloader

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss']
                                     for x in val_step_outputs]).mean()
        return {'val_loss': avg_val_loss}


trainer = pl.Trainer(progress_bar_refresh_rate=30, gpus=1, max_epochs=200)
trainer.fit(CaptchaModel())
