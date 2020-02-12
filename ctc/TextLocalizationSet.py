import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re


# This function sorts the files when using os.listdir()
# because os.listdir() returns without order
# I grabbed it online let me know if this becomes a problem
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


# custom dataset class for text localization dataset (Task4.1 of ICDAR 2015)
class TextLocalizationSet(Dataset):

    def __init__(self, img_dir, gt_dir, transform=None):
        '''Class Attributes:
            self.img_dir: is the directory name with all the images
            self.gt_dir: is the directory name with all the ground truth text files (gt stands for ground truth)
            self.img_dirs: is the list of all the image names
            self.gt_dirs: is the list of all the gt text file names
            self.transform: is the transform performed on the dataset
        '''
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_dirs = sorted_alphanumeric(os.listdir(img_dir))
        self.gt_dirs = sorted_alphanumeric(os.listdir(gt_dir))
        self.transform = transform

    # returns the size of the dataset ==> number of images
    def __len__(self):
        return len(self.img_dirs)

    # allows indexing of the dataset ==> ex. dataset[0] returns a single sample from the dataset
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        # path to image
        img_name = os.path.join(self.img_dir, self.img_dirs[item])
        image = io.imread(img_name)

        # path to ground truth text file
        gt_dir = os.path.join(self.gt_dir, self.gt_dirs[item])

        # read the ground truth
        with open(gt_dir, "r", encoding='utf-8-sig') as infile:
            gts_file = infile.readlines()
            # U+FEFF is the Byte Order Mark character, which occurs at the start of a document
            gts_file[0] = gts_file[0].lstrip("\ufeff")

        # empty list where 0 as a placeholder
        coords_list = []
        transcriptions = []

        # dictionary containing all the ground truths
        # remember each word in the image has a ground truth (coordinates and transcription)
        for i, gt in enumerate(gts_file):
            gt = gt.split(",")
            gt[-1] = gt[-1].rstrip()

            # Remove entries with ### since it is invalid
            if gt[-1] == '###':
            	continue

            # make 2D list where each row is a coordinate in this form [x, y]
            coords = np.array(gt[:-1], dtype=int)
            size = 4
            coordsList = [0 for _ in range(size)]
            for j in range(size):
                coordsList[j] = [coords[j*2], coords[j*2 + 1]]

            coords_list.append(np.array(coordsList))
            transcriptions.append(gt[-1])

        # perform transforms
        if self.transform:
            image = self.transform(image)

        return image, coords_list, transcriptions


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        for i in range(len(gt)):
            gt[i]["coords"] = gt[i]["coords"] * [new_w / w, new_h / h]

        return {'image': img, 'gt': gt}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        for i in range(len(gt)):
            gt[i]["coords"] = torch.from_numpy(gt[i]["coords"])

        return {'image': torch.from_numpy(image),
                'gt': gt}


if __name__ == "__main__":
    # example usage
    dataset = TextLocalizationSet(img_dir="train",
                                  gt_dir="gt")

    # can use DataLoader now with this custom dataset


    ''' sample will be a dictionary containing two keys: "image" and "gt" (denoting ground truth)
        The value to "image" will be just ndarray (n dimensional array) of the image
        The value to "gt" will be a list of dictionary again, one dictionary per word in the image.
        These dictionaries have two keys: "coords" and "transcription".
        The value to "coords" is a 2D nparray with coordinates in this form: [[x1, y1], ..., [x4, y4]] 
        We have 8 coordinates denoting the bounding box beginning from x1,y1 to x4,y4 in clockwise.
        The value to "transcription" is a string that is the actual word to be recognized in the image.
    '''

    # example usage
    dataset = TextLocalizationSet(img_dir="train",
                                  gt_dir="gt",
                                  transform=transforms.Compose([Rescale(256),
                                                                ToTensor()]))

    # can use DataLoader now with this custom dataset
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=4)
