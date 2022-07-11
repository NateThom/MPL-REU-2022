import pandas as pd
import torch
from math import floor
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt


class Occluded_Dataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
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
        for f in range(32):
            folder = '{0:05b}'.format(f)+'/'
            for im in range(split_min, split_max):
                self.image_names.append(path+folder+str(im).zfill(6)+'.jpg')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.image_names[idx]
        image = torchvision.io.read_image(img_path)
        image = TF.convert_image_dtype(image, torch.float)
        gender = self.labels[int(self.image_names[idx][-10:-4])]
        label = torch.tensor([1-gender, gender])
        return image, label


class CelebA_Dataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        attr_df = pd.read_csv('../../../CelebA/Anno/list_attr_celeba.csv')
        img_names = attr_df.image_name.values.tolist()
        img_names = [int(v[:-4]) for v in img_names]
        img_gender = attr_df.Male.values.tolist()
        img_gender = [int(v > 0) for v in img_gender]
        self.labels = dict(zip(img_names, img_gender))
        path = '../../Data_Augmentation/IMG_HiRes/'
        numImgs = 2048
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
        image = TF.convert_image_dtype(image, torch.float)
        gender = self.labels[int(self.image_names[idx][-10:-4])]
        label = torch.tensor([1-gender, gender])
        return image, label


def show_batch(b):
    labels_map = {0:'female', 1:'male'}
    imgs, lbls = b
    fig = plt.figure()
    cols, rows = 4, 4
    for i in range(16):
        img, lbl = imgs[i], lbls[i]
        fig.add_subplot(rows, cols, i+1)
        plt.title(labels_map[int(lbl)], fontsize=10, y=0.92)
        plt.axis("off")
        plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.show()


# occluded_dataset = Occluded_Dataset((0, 0.001))
# dataloader = DataLoader(occluded_dataset, batch_size=16, num_workers=8, drop_last=False, shuffle=True)
#
# for batch in dataloader:
#    show_batch(batch)
