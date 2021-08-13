import os
import csv
import torch
import numpy as np
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils import processImg


class AdultDataset(Dataset):
    def __init__(self, annotation_file, device='cpu'):
        super(AdultDataset, self).__init__()
        self.device = device
        self.data_list = []
        exceptions = 0
        self.n_adult = 0
        self.n_notAdult = 0

        with open(annotation_file) as csv_file:
            reader = csv.reader(csv_file, delimiter=';')
            line_count = 0

            for row in reader:
                if line_count == 0:
                    line_count = 1
                else:
                    try:
                        _ = Image.open(row[0])
                    except:
                        exceptions += 1
                        continue

                    if int(row[1]) > 0:
                        self.n_adult += 1
                    else:
                        self.n_notAdult += 1

                    self.data_list.append(row)

        print("Exceptions: ", exceptions)
        self.dict_numbers = {'0': self.n_notAdult, '1': self.n_adult}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        label = int(data[1])

        raw_img = Image.open(data[0])
        img = processImg(raw_img, resize=True)

        img = torchvision.transforms.ToTensor()(img)

        if len(img.shape) < 3:
            img = img.unsqueeze(0)

        return img.to(self.device), torch.tensor(label).unsqueeze(0).to(self.device)


def datasetLoader(annotation_file, batch_size=32, shuffle=True, device='cpu'):
    dataset = AdultDataset(annotation_file=annotation_file, device=device)

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle)

    return loader, dataset


if __name__ == "__main__":
    data_loader, _ = datasetLoader(r'./data/UTKFace/adult_annotation.csv')

    for imgs, labels in data_loader:
        print(imgs.shape)
        print(labels.shape)
        print("Max: ", imgs[0, 0, :, :].max())
        print("Min: ", imgs[0, 0, :, :].min())
        print("Label", labels[0])

        plt.imshow(imgs[0, 0, :, :], cmap='gray')
        plt.show()
