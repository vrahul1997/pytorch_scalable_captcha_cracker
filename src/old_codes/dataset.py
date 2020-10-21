import albumentations
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset(Dataset):
    def __init__(self, image_paths, targets, resize=None):
        self.images = image_paths
        self.targets = targets
        self.resize = resize

        self.aug = albumentations.Compose(
            [albumentations.Normalize(
                always_apply=True)
             ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item]).convert("RGB")
        target = self.targets[item]

        # Pil accepts resize in the width first approach, so when resizing the image width should be first and height should be second
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        # convert the images into numpy array before applying the augmentations
        image = np.array(image)
        aug_image = self.aug(image=image)
        image = aug_image["image"]

        # we should transpose these numpy arrays into torch versions of transposed images
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # convert outputs into tensor
        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.long),
        }
