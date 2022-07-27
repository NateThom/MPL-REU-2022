import pandas as pd
import torch
from math import floor
import random
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as tf
import matplotlib.pyplot as plt

class OccludedDataset(Dataset):
    def __init__(self, split, transform=None):
        augs = transform
        attr_df = pd.read_csv('../../../CelebA/Anno/list_attr_celeba.csv')
        img_names = attr_df.image_name.values.tolist()
        img_names = [int(v[:-4]) for v in img_names]
        img_gender = attr_df.Male.values.tolist()
        img_gender = [int(v > 0) for v in img_gender]
        self.labels = dict(zip(img_names, img_gender))
        path = '../../Data_Augmentation/augmented_data/'
        self.image_names = []
        split_min = floor(split[0] * 202599) + 1
        split_max = floor(split[1] * 202599) + 1
        folder = ''.join([str(a) for a in augs]) + '/'
        for im in range(split_min, split_max):
            self.image_names.append(path+folder+str(im).zfill(6)+'.jpg')
        for im in range(split_min, split_max):
            self.image_names.append(path+'00000/'+str(im).zfill(6)+'.jpg')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.image_names[idx]
        image = torchvision.io.read_image(img_path)
        image = tf.convert_image_dtype(image, torch.float)
        gender = self.labels[int(self.image_names[idx][-10:-4])]
        label = torch.tensor([1-gender, gender])
        return image, label


class CelebADataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        attr_df = pd.read_csv('../../../CelebA/Anno/list_attr_celeba.csv')
        img_names = attr_df.image_name.values.tolist()
        img_names = [int(v[:-4]) for v in img_names]
        img_gender = attr_df.Male.values.tolist()
        img_gender = [int(v > 0) for v in img_gender]
        self.labels = dict(zip(img_names, img_gender))
        path = '../../Data_Augmentation/IMG_HiRes/'
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
        if self.transform == 'test':
            return image, label, img_path[-10:]
        return image, label


class LFWDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        attr_df = pd.read_csv('../../../LFW/lfw_gender_labels.csv')
        img_names = attr_df.image_name.values.tolist()
        img_names = [int(v[:-4]) for v in img_names]
        img_gender = attr_df.gender.values.tolist()
        img_gender = [int(v > 0) for v in img_gender]
        self.labels = dict(zip(img_names, img_gender))
        path = '../../../LFW/IMG_numbered/'
        self.image_names = []
        split_min = floor(split[0] * 13233) + 1
        split_max = floor(split[1] * 13233) + 1
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
        if self.transform == 'test':
            return image, label, img_path[-10:]
        return image, label


class HEATDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        attr_df = pd.read_csv('../../../HEAT/Anno/heat_gender_labels.csv')
        img_names = attr_df.image_name.values.tolist()
        img_names = [int(v[:-4]) for v in img_names]
        img_gender = attr_df.gender.values.tolist()
        img_gender = [int(v > 0) for v in img_gender]
        self.labels = dict(zip(img_names, img_gender))
        path = '../../../HEAT/Img/'
        self.image_names = []
        for im in range(1, 8279):
            self.image_names.append(path+str(im).zfill(4)+'.jpg')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.image_names[idx]
        image = torchvision.io.read_image(img_path)
        image = tf.convert_image_dtype(image, torch.float)
        gender = self.labels[int(self.image_names[idx][-8:-4])]
        label = torch.tensor([1-gender, gender])
        if self.transform == 'test':
            return image, label, img_path[-8:]
        return image, label


# class OccludedDataset(Dataset):
#     def __init__(self, split, transform=None):
#         augs = transform[0]
#         aug_sets = (2 ** len([i for i in augs if i == 1])) - 1
#         portion = transform[1]
#         attr_df = pd.read_csv('../../../CelebA/Anno/list_attr_celeba.csv')
#         img_names = attr_df.image_name.values.tolist()
#         img_names = [int(v[:-4]) for v in img_names]
#         img_gender = attr_df.Male.values.tolist()
#         img_gender = [int(v > 0) for v in img_gender]
#         self.labels = dict(zip(img_names, img_gender))
#         path = '../../Data_Augmentation/augmented_data/'
#         self.image_names = []
#         split_min = floor(split[0] * 202599) + 1
#         split_max = floor(split[1] * 202599) + 1
#         total = 2 * (split_max-split_min)
#         augmented = int(total * portion) // aug_sets
#         augmented += (0, 1)[augmented < (split_max - split_min)]
#         unaugmented = int(total - (augmented * aug_sets))
#         random.seed(a=128)
#         surplus_aug = (augmented * aug_sets) - total
#         ticker = 0
#         for f in range(1, 32):
#             folder = '{0:05b}'.format(f)+'/'
#             selected = True
#             for a in range(5):
#                 if folder[a] == '1' and augs[a] == 0:
#                     selected = False
#             if selected:
#                 ticker += 1
#                 comp = 1 if surplus_aug > 0 and ticker <= surplus_aug else 0
#                 for im in random.sample(range(split_min, split_max), augmented - comp):
#                     self.image_names.append(path+folder+str(im).zfill(6)+'.jpg')
#         for im in random.sample(range(split_min, split_max), max(unaugmented, 0)):
#             self.image_names.append(path+'00000/'+str(im).zfill(6)+'.jpg')

def showbatch(b):
    labels_map = {0:'female', 1:'male'}
    imgs, lbls = b
    fig = plt.figure()
    cols, rows = 4, 4
    for i in range(16):
        img, lbl = imgs[i], lbls[i]
        lbl = list(lbl).index(1)
        fig.add_subplot(rows, cols, i+1)
        plt.title(labels_map[int(lbl)], fontsize=10, y=0.92)
        plt.axis("off")
        plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.show()

def makesample():
    celeba_dataset = CelebADataset((0, 0.001))
    dataloader = DataLoader(celeba_dataset, batch_size=16, num_workers=8, drop_last=False, shuffle=True)

    for batch in dataloader:
        showbatch(batch)
