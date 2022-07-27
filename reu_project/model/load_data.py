import pandas as pd
import torch
from math import floor
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as tf
from torchvision import transforms
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


class StandardAugDataset(Dataset):
    def __init__(self, split, transform=None):
        tfs = transform
        tf_dict = {0: transforms.RandomRotation(90),
                   1: transforms.RandomErasing(),
                   2: transforms.GaussianBlur(9),
                   3: transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.2),
                   4: transforms.RandomResizedCrop(224)}
        self.transform = transforms.Compose([tf_dict[t] for t in range(5) if tfs[t] == 1])
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
        for i in range(2):
            for im in range(split_min, split_max):
                self.image_names.append(path+str(im).zfill(6)+'.jpg')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.image_names[idx]
        image = torchvision.io.read_image(img_path)
        image = tf.convert_image_dtype(image, torch.float)
        if idx >= len(self.image_names) / 2:
            image = self.transform(image)
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


class HEADDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        attr_df = pd.read_csv('../../../HEAD/Anno/head_gender_labels.csv')
        img_names = attr_df.image_name.values.tolist()
        img_names = [int(v[:-4]) for v in img_names]
        img_gender = attr_df.gender.values.tolist()
        img_gender = [int(v > 0) for v in img_gender]
        self.labels = dict(zip(img_names, img_gender))
        path = '../../../HEAD/Img/'
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
    celeba_dataset = StandardAugDataset((0, 0.0005), transform=[1, 1, 1, 1, 1])
    dataloader = DataLoader(celeba_dataset, batch_size=16, num_workers=8, drop_last=False, shuffle=True)

    for batch in dataloader:
        showbatch(batch)
        pass
