
import torch
import cv2
import numpy
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from PIL import Image


class PASCAL_VOL2012(Dataset):
    def  __init__(self, data_path, txt_path, transform):
        self.data_path = data_path
        self.txt_path = txt_path
        self.transform = transform
        with open(txt_path, 'r') as input_file:
            lines = input_file.readlines()
        input_file.close()
        self.img_names = [line.split(' ')[0] for line in lines]
        self.str_labels = [line.split(' ')[1:] for line in lines]
        for in_list in self.str_labels:
            for idx, item in enumerate(in_list):
                in_list[idx] = int(item)

    def __getitem__(self, index):

        imagepath = os.path.join(self.data_path, f"{self.img_names[index][:-4]+'.jpg'}")
        image_read = Image.open(imagepath).convert('RGB')
        image = self.transform(image_read)
        targets = self.str_labels[index]

        return {
            'image': image,
            'label': torch.tensor(targets, dtype=torch.int64),
            'image_name':self.img_names[index],
            'image_path':imagepath
        }

    def __len__(self):
        return len(self.img_names)