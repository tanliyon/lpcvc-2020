import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from groundtruth import *
import re


def sorted_alphanumeric(data):
    """
    This function sorts the files when using os.listdir()
    because os.listdir() returns without order
    :param data: list of strings to be sorted
    :return: sorted list of strings
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def collate(batch):
    name = [item["name"] for item in batch]
    image = [item["image"] for item in batch]
    coords = [item["coords"] for item in batch]
    transcriptions = [item["transcription"] for item in batch]
    return [name, image, coords, transcriptions]


class TextLocalizationSet(Dataset):
    """ Custom dataset class for text localization dataset (Task4.1 of ICDAR 2015)
    :attr self.img_dir: directory name with all the images
    :attr self.gt_dir: directory name with all the ground truth text files (gt stands for ground truth)
    :attr self.img_dirs: list of all the image names
    :attr self.gt_dirs: list of all the gt text file names
    :attr self.transform: transform performed on the dataset
    """

    def __init__(self, image_directory_path, annotation_directory_path, new_length, transform=None):
        """
        :param train: boolean indicating if the data to be loaded is train or test set
        :param transform: transform performed on the dataset (default=None)
        """

        self.img_dir = image_directory_path
        self.gt_dir = annotation_directory_path
        self.img_dirs = sorted_alphanumeric(os.listdir(self.img_dir))
        self.gt_dirs = sorted_alphanumeric(os.listdir(self.gt_dir))
        self.new_length = new_length
        self.transform = transform

    def __len__(self):
        """
        :return: size of the dataset (aka number of images)
        """
        return len(self.img_dirs)

    def __getitem__(self, idx):
        """
        Allows indexing of the dataset ==> ex. dataset[0] returns a single sample from the dataset
        :param idx: index value of dataset sample we want to return
        :return: dictionary with three keys: "image", "coords" and "transcription"
                 value to "image": ndarray of pixel values
                 value to "coords": 2d array of x, y coordinates (i.e [ [x1, y1]...[x4, y4]])
                 value to "transcription": string denoting actual word in image
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read image
        img_name = os.path.join(self.img_dir, self.img_dirs[idx])
        image = io.imread(img_name)

        # path to ground truth text file
        gt_dir = os.path.join(self.gt_dir, self.gt_dirs[idx])

        # read the ground truth
        with open(gt_dir, "r", encoding='utf-8-sig') as infile:
            gt_from_file = infile.readlines()
            # U+FEFF is the Byte Order Mark character, which occurs at the start of a document
            gt_from_file[0] = gt_from_file[0].lstrip("\ufeff")

        # initialize empty lists to hold final data
        total_coords = []
        transcriptions = []
        bool_tags = []

        # dictionary containing all the ground truths
        # each word in the image has a ground truth (coordinates and transcription)
        for i, one_gt in enumerate(gt_from_file):
            one_gt = one_gt.split(",")
            one_gt[-1] = one_gt[-1].rstrip("\n")

            # Remove entries with ### since they are invalid
            if one_gt[-1] == "###" or one_gt[-1] == "*":
                bool_tags.append(True)
            else:
                # remove non-alphanumeric characters
                pattern = re.compile(r"[^\w\d\s]")
                one_gt[-1] = re.sub(pattern, "", one_gt[-1])

                # if transcription is empty after removing non-alphanumerics, skip
                if one_gt[-1] == "":
                    continue

                bool_tags.append(False)

            # arrange coordinates into 2d array form [ [x1, y1] ... [x4, y4] ]
            coords = one_gt[:-1]
            size = 4
            single_coords = [[] for _ in range(size)]
            for j in range(0, 2 * size, 2):
                single_coords[j // 2] = [coords[j], coords[j + 1]]

            total_coords.append(np.array(single_coords, dtype=int))

            transcriptions.append(one_gt[-1])

        sample = {'name': img_name, 'image': image, 'coords': np.array(total_coords, dtype=np.float32), 'transcription': transcriptions, 'bool': np.array(bool_tags, dtype=np.bool)}

        rescale = Rescale((self.new_length, self.new_length))
        sample = rescale.__call__(sample)

        groundtruth = GroundTruthGeneration(sample['name'], sample['image'])
        groundtruth.Load_Geometry_Score_Maps(np.array(sample['coords'], dtype=np.float32), sample['bool'])

        # perform transforms
        if self.transform:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
            sample['image'] = self.transform(sample['image'])

        return sample


class Rescale(object):
    """ Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, coords = sample['image'], sample['coords']

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
        for i in range(len(coords)):
            coords[i] = np.array(coords[i], dtype=float) * [new_w / w, new_h / h]
            coords[i] = coords[i].round(0)
            coords[i] = np.array(coords[i], dtype=int)

        sample["image"] = img
        sample["coords"] = coords

        return sample


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors. """

    def __call__(self, sample):
        image, coords = sample['image'], sample['coords']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        for i in range(len(coords)):
            coords[i] = torch.from_numpy(coords[i])

        sample["image"] = torch.from_numpy(image)
        sample["coords"] = coords

        return sample


if __name__ == "__main__":
    imagePath = "./Dataset/Train/TrainImages/"
    annotationPath = "./Dataset/Train/TrainTruth/"

    # example usage
    dataset = TextLocalizationSet(imagePath, annotationPath, 256)

    # print(dataset[0])

    # can use DataLoader now with this custom dataset
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=4,
                             collate_fn=collate)

    for data in data_loader:
        print(data[2])
        break
