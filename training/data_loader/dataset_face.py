import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image 

class FaceDataset(Dataset):
    def __init__(self, img_path, mask_path, resolution=512, same_prob=0.1):
        self.resolution = resolution
        self.same_prob = same_prob
        self.img_path = img_path 
        self.mask_path = mask_path
        self.files = os.listdir(self.img_path)
        self.length = len(self.files)
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transforms_mask = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        target = self.transforms(Image.open(self.img_path + "/" + self.files[index]).resize([self.resolution, self.resolution]))
        mask = self.transforms_mask(Image.open(self.mask_path + "/" + self.files[index]).resize([self.resolution, self.resolution]))
        if np.random.uniform() < self.same_prob:
            source = Image.open(self.img_path + "/" + self.files[index]).resize([self.resolution, self.resolution])
            source = self.transforms(source)
            same = torch.tensor(1.)
        else:
            rand_idx = np.random.randint(0, self.length)
            source = Image.open(self.img_path + "/" + self.files[rand_idx]).resize([self.resolution, self.resolution])
            source = self.transforms(source)
            same = torch.tensor(0.)  

        mask = (mask > 0.5).float()

        return target, source, mask[0:1], same

# from tqdm import tqdm 
# path = "/data1/GMT/Dataset/thumbnails128x128/"
# files = os.listdir(path)
# for idx, file in tqdm(enumerate(files)):
#     img = cv2.imread(path + file)
#     img = img*2
