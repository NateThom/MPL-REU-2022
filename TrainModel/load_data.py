import glob
import os
import random
import pandas as pd
import torch
from math import floor
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import functional as tf


class AugmentedDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        attr_df = pd.read_csv('list_attr_celeba.csv')

        # get names, gender and create labels
        img_names = attr_df.image_name.values.tolist()
        img_names = [int(v[:-4]) for v in img_names]
        img_gender = attr_df.Male.values.tolist()
        img_gender = [int(v > 0) for v in img_gender]
        self.labels = dict(zip(img_names, img_gender))

        # add image names with the entire path to them
        path = '/home/guest/MPL-REU-2022/swapped_attributes/eyes/'
        os.chdir(path)
        self.image_names = []
        self.image_names_ref = {}

        split_min = floor(split[0] * 202599) + 1
        split_max = floor(split[1] * 202599) + 1
        for im in range(split_min, split_max):
            augmented_image = []
            # find all images corresponding to the
            for file in glob.glob(path + str(im) + '/*.jpg'):
                augmented_image.append(file)
            self.image_names_ref[im] = augmented_image
            self.image_names.append(im)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        index = self.image_names[idx]
        possible_images = self.image_names_ref[index]
        img_path = possible_images[random.randint(0, len(possible_images) - 1)]
        image = torchvision.io.read_image(img_path)
        image = tf.convert_image_dtype(image, torch.float)
        gender = self.labels[int(img_path[-10:-4])]
        return image, gender


class CelebADataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        attr_df = pd.read_csv('list_attr_celeba.csv')
        img_names = attr_df.image_name.values.tolist()
        img_names = [int(v[:-4]) for v in img_names]
        img_gender = attr_df.Male.values.tolist()
        img_gender = [int(v > 0) for v in img_gender]
        self.labels = dict(zip(img_names, img_gender))
        path = '/home/guest/MPL-REU-2022/img_high_res/'
        self.image_names = []
        split_min = floor(split[0] * 202599) + 1
        split_max = floor(split[1] * 202599) + 1
        for im in range(split_min, split_max):
            self.image_names.append(path+str(im).zfill(6)+'.jpg')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.image_names[idx]
        image = torchvision.io.read_image(img_path)
        image = tf.convert_image_dtype(image, torch.float)
        gender = self.labels[int(self.image_names[idx][-10:-4])]
        label = torch.tensor([1-gender, gender])
        return image, label


# This new version of the data loader takes split as a parameter. split is a 2-tuple
# containing two floats between 0 and 1. Instead of iterating through a predefined
# numImg, the function finds the subset of the dataset defined by split (lines 21 and 22),
# and iterates through this subset. This allows a dataset to be split for
# training/validation/testing. CelebA is split by default 80%/10%/10%.
