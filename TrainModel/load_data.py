import os
import random
import pandas as pd
import torch
from math import floor
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import functional as tf
import glob


class AugmentedDataset(Dataset):
    def __init__(self, split, transform=None):
        self.image_augmentations = ['eyebrows', 'eyes', 'nose', 'mouth', 'chin', 'jaw', 'cheek']
        self.transform = transform
        attr_df = pd.read_csv('list_attr_celeba.csv')

        # get names, gender and create labels
        img_names = attr_df.image_name.values.tolist()
        img_names = [int(v[:-4]) for v in img_names]
        img_gender = attr_df.Male.values.tolist()
        img_gender = [int(v > 0) for v in img_gender]
        self.labels = dict(zip(img_names, img_gender))

        # add image names
        path = '/home/guest/MPL-REU-2022/swapped_attributes/'
        self.image_names = []

        split_min = floor(split[0] * 202599) + 1
        split_max = floor(split[1] * 202599) + 1

        # randomly choose which images in the split will be augmented
        total_augmented = int(transform[1] * (split_max - split_min))
        random.seed(a=128)
        chosen_augmented = random.sample(range(split_min, split_max), total_augmented)
        indices_aug = [x for x, y in enumerate(transform[0]) if y != 0]
        list_augmentations = []
        for t in range(split_min, split_max):
            if t in chosen_augmented:
                list_augmentations.append(random.choice(indices_aug))
            else:
                list_augmentations.append(0)

        # add image names to list
        for im in range(split_min, split_max):
            os.chdir(path)

            if im in chosen_augmented:
                # get path to directory for image numbered im
                curr_path = path + str(self.image_augmentations[list_augmentations[im - split_min]]) + '/' + str(im) + '/'
                os.chdir(curr_path)
                list_augmented_images = glob.glob(curr_path + '/*.jpg')

                # choose random image from im directory, add to list
                if not list_augmented_images:
                    print(curr_path)
                else:
                    self.image_names.append(random.choice(list_augmented_images))

            else:
                self.image_names.append(path + 'normal/' + str(im).zfill(6) + '.jpg')

        os.chdir('/home/guest/MPL-REU-2022/TrainModel/')


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.image_names[idx]
        image = torchvision.io.read_image(img_path)
        image = tf.convert_image_dtype(image, torch.float)
        gender = self.labels[int(self.image_names[idx][-10:-4])]
        label = torch.tensor([1 - gender, gender])
        if self.transform == 'test':
            return image, label, img_path[-10:]

        return image, label


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
            self.image_names.append(path + str(im).zfill(6) + '.jpg')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.image_names[idx]
        image = torchvision.io.read_image(img_path)
        image = tf.convert_image_dtype(image, torch.float)
        gender = self.labels[int(self.image_names[idx][-10:-4])]
        label = torch.tensor([1 - gender, gender])
        if self.transform == 'test':
            return image, label, img_path[-10:]

        return image, label

# This new version of the data loader takes split as a parameter. split is a 2-tuple
# containing two floats between 0 and 1. Instead of iterating through a predefined
# numImg, the function finds the subset of the dataset defined by split (lines 21 and 22),
# and iterates through this subset. This allows a dataset to be split for
# training/validation/testing. CelebA is split by default 80%/10%/10%.
