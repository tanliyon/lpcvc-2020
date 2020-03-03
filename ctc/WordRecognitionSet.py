import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
from PIL import Image


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


def load_coords(filename):
    """
    This function loads data from ground truth coordinates data from text files
    :param filename: name of the file holding the data
    :return: return extracted data as either a list of list
    """
    result = []
    with open(filename, "r") as infile:
        file_data = infile.readlines()

    for i in range(len(file_data)):
        new_item = file_data[i].split(",")[1:]  # first item is always image name so take out
        result.append(new_item)
    return result


def load_transcription(filename):
    """
    This function loads data from ground truth transcription from text files
    :param filename: name of the file holding the data
    :return: return extracted transcription as list of strings
    """
    result = []
    with open(filename, "r") as infile:
        file_data = infile.readlines()

    t_pattern = re.compile(r"(?P<transcription>(?<=\")(.)*(?=\"))")

    for i in range(len(file_data)):
        new_item = file_data[i].split(", ")[1:]  # first item is always image name so take out
        new_item = new_item[0]
        new_item = new_item.strip("\n")

        # get transcription
        match = re.search(t_pattern, file_data[i])
        new_item = match["transcription"]

        # remove non-alphanumeric characters
        non_alphanum = re.compile(r"[^\w\d\s]")
        new_item = re.sub(non_alphanum, "", new_item)

        result.append(new_item)
    return result


def collate(batch):
    image = [item["image"] for item in batch]
    transcriptions = [item["transcription"] for item in batch]
    return [image, transcriptions]


class WordRecognitionSet(Dataset):
    """ Custom dataset class for Word Recognition dataset (Task4.1 of ICDAR 2015)
    :attr self.which_set: boolean indicating if the data to be loaded is train or test set
    :attr self.img_dir: directory name with all the images
    :attr self.gt_dir: directory name with all the ground truth text files (gt stands for ground truth)
    :attr self.img_dirs: list of all the image names
    :attr self.gt_list: ordered list of ground truths
    :attr self.coords_list: ordered list of ground truth coordinates
    :attr self.transform: transform performed on the dataset
    """

    def __init__(self, train, transform=None):
        """
        :param train: boolean indicating if the data to be loaded is train or test set
        :param transform: transform performed on the dataset (default=None)
        """

        self.which_set = "train" if train else "test"
        self.img_dir = self.which_set
        self.gt_dir = self.which_set + "_gt"
        self.img_dirs = sorted_alphanumeric(os.listdir(self.img_dir))
        self.gt_list = load_transcription(self.gt_dir + "/gt.txt")
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
        :return: dictionary with two keys: "image" and "gt"
                 value to "image": ndarray of pixel values
                 value to "gt": dictionary with two keys: "coords" and "transcription"
                                value to "coords": 2d array of x, y coordinates (i.e [ [x1, y1]...[x4, y4]])
                                value to "transcription": string denoting actual word in image
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # path to image
        img_name = os.path.join(self.img_dir, self.img_dirs[idx])
        image = Image.open(img_name)

        sample = {'image': image,
                  "transcription": self.gt_list[idx]}

        # perform transforms
        if self.transform:
            sample['image'] = self.transform(image)

        return sample


if __name__ == "__main__":
    # example usage
    dataset = WordRecognitionSet(train=True)

    #print(dataset[0])

    # can use DataLoader now with this custom dataset
    data_loader = DataLoader(dataset, batch_size=4,
                             shuffle=False, num_workers=4,
                             collate_fn=collate)

    for data in data_loader:
        print(data)
        break
