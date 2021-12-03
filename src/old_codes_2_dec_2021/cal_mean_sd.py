import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import glob
from sklearn import preprocessing, metrics, model_selection
import config
import dataset

# image_files
image_files = glob.glob("../input/train_all_captchas/*.png")
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

def get_mean_std(loader):

    # variance(x) = Expected(x**2) - Expected(x)**2

    channels_sum, channels_squarred_sum, num_batches = 0, 0, 0

    for data in train_dataloader:
        images = data["images"]
        channels_sum += torch.mean(data["images"], dim=[0, 2, 3])
        channels_squarred_sum += torch.mean(data["images"]**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squarred_sum/num_batches - mean**2)**0.5

    return mean, std

mn, st = get_mean_std(train_dataloader)
print(mn, st)

