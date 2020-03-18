# encoding: utf-8
#  -*- encoding: utf-8 -*-
import os
import torch
import numpy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import xml.etree.ElementTree as ET
# from utils.config import CONFIG
# from skimage import io, transform
import random
import re

# class textDataset(Dataset):
#     def __init__(self, xml_file, root_dir, img_dir,  transform=None):
#         self.xml_file = xml_file
#         self.root_dir = root_dir
#         self.transform = transform
#         self.img_dir = img_dir
#         self.tree = ET.parse(os.path.join(self.root_dir, self.xml_file))
#
#     def __len__(self):
#         return(len(self.tree.getroot().getchildren()) - 1)
#         # return len(self.)
#     def __getitem__(self, idx):
#         image = Image.open(os.path.join(self.root_dir, self.img_dir, str(idx + 1) + ".jpg"))
#         img = self.transform(image)
#         image.close()
#         text = self.tree.getroot()[idx + 1].attrib["tag"]
#         return(img, text)
#


class syn_text(Dataset):
    def __init__(self, anno_file, root_dir,  transform=None):
        self.anno_file = anno_file
        self.root_dir = root_dir
        self.transform = transform
        self.gt = self.manage_gt(self.anno_file)

    def __len__(self):
        return len(self.gt)
        # return len(self.)

    def manage_gt(self, gt_file):
        with open(os.path.join(self.root_dir, self.anno_file)) as file:
            lines = file.readlines()
            return lines

    def __getitem__(self, idx):
        try:
            paths = self.gt[idx].split("./")[1].split(" ")[0].split("/")
            final_path = self.root_dir
            for path in paths:
                final_path = os.path.join(final_path, path)

            image = Image.open(final_path)
            if self.transform:
            	img = self.transform(image)
            image.close()
            text = self.gt[idx].split("_")[1]
        except Exception as e:
            print(e)
            print("error loading image or text loading error")
            img, text = self.__getitem__(random.randint(1, len(self.gt) - 1))
        return (img, text)



class textDataset(Dataset):
    def __init__(self, text_file, root_dir, img_dir, phase,  transform=None):
        self.text_file = text_file
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = img_dir
        self.gt = self.manage_gt(self.text_file)
        self.phase = phase

    def __len__(self):
        return len(self.gt)
        # return len(self.)

    def manage_gt(self, gt_file):
        with open(os.path.join(self.root_dir, self.text_file)) as file:
            lines = file.readlines()
            return lines

    def __getitem__(self, idx):
        mode = int(idx)
        if self.phase == "val":
            mode = 3999
            mode = mode + idx
        image = Image.open(os.path.join(self.root_dir, self.img_dir, "word_" + str(mode + 1) + ".png"))
        img = self.transform(image)
        image.close()
        replace_sale = ["Sale", "SALE", "sale"]
        try:
            text = re.findall(r'\"(.+?)\"', self.gt[idx])[0]
        except:
            text = ""
        if("s" in text.lower() and random.random() < 0.9):
            img, text =  self.__getitem__(random.randint(1, len(self.gt) - 1))
        return (img, text)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST), transforms.ToTensor()])
    dataset = syn_text("annotation_train.txt", "/home/damini/Downloads/mnt/ramdisk/max/90kDICT32px/", "train", transform)
    print(dataset.__getitem__(0))
    train_loader = DataLoader(dataset, batch_size= CONFIG.BATCH_SIZE, shuffle=True)
    image = next(iter(train_loader))
    print("done")
