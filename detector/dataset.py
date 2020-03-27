import os
import re
import torch
import numpy as np
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from groundtruth import *
from detect import *

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
    path = [item["path"] for item in batch]
    image = [item["image"] for item in batch]
    coords = [item["coords"] for item in batch]
    transcriptions = [item["transcription"] for item in batch]
    score = [item["score"] for item in batch]
    mask = [item["mask"] for item in batch]
    geometry = [item["geometry"] for item in batch]
    return [name, path, image, coords, transcriptions, score, mask, geometry]


class TextLocalizationSet(Dataset):
    def __init__(self, image_directory_path, annotation_directory_path, new_dimensions, transform=True):
        self.image_directory_path = image_directory_path
        self.annotation_directory_path = annotation_directory_path
        self.image_paths, self.image_names, self.annotation_paths = self.Get_Images_Path_Name_Annotation()
        self.new_dimensions = new_dimensions
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))])

        image_path = os.path.join(self.image_directory_path, self.image_paths[idx])
        image = self.Load_Image(image_path)
        quad_coordinates, text_tags, bool_tags = self.Load_Annotation(os.path.join(self.annotation_directory_path, self.annotation_paths[idx]))

        sample = {'name': self.image_names[idx], 'path': image_path, 'image': tf(Image.fromarray(image)), 'coords': np.array(quad_coordinates, dtype=np.float32), 'transcription': text_tags, 'score': None, 'mask': None, 'geometry': None}

        rescale = Rescale(self.new_dimensions)
        sample = rescale.__call__(sample)

        groundtruth = GroundTruthGeneration(sample['name'], sample['image'])
        sample['score'], sample['mask'], sample['geometry'] = groundtruth.Load_Geometry_Score_Maps(sample['coords'], bool_tags)

        #if self.transform:
        #
        # sample['image'] = tf(image)

        return sample

    def Get_Images_Path_Name_Annotation(self, mode="training"):
        """
        :description: Gets all image paths and image names from train or test directory

        :return:      image_paths (list): sorted image paths in directory
                      image_names (list): sorted image names in directory
        """
        does_directory_exist(self.image_directory_path)

        image_paths = os.listdir(self.image_directory_path)
        is_valid_image_file(image_paths)

        annotation_paths = os.listdir(self.annotation_directory_path)
        is_valid_text_file(annotation_paths)

        image_names = []
        for i in range(0, len(image_paths)):
            image_names.append(image_paths[i].split("/")[-1])

        #logging.info("\nPreparing {} images for {}\n".format(len(image_paths), mode))
        return sorted_alphanumeric(image_paths), sorted_alphanumeric(image_names), sorted_alphanumeric(annotation_paths)

    def Load_Image(self, image_path):
        """
        :description: Loads the image from image file and makes it a numpy array

        :param:       image_path (str): path of the image file in question

        :return:      image (uint8 numpy array): image data (channel, height, width)
        """
        #image = np.array(Image.open(image_path))    # type -> uint8
        #return np.moveaxis(image, 2, 0)             # reverses order -> channel x height x width
        image = io.imread(image_path, as_gray=True)
        return image

    def Load_Annotation(self, annotation_file):
        """
        :description: Loads the ground truth coordinates of the text box and text labels from annotation file of image

        :param:       image_name (str): name of the image in question

        :return:      quad_coordinates (float32 numpy array): 8 coordinates of ground truth box
                      text_tags (bool numpy array): ground truth text labels
        """
        quad_coordinates = []
        text_tags = []
        bool_tags = []

        with open(annotation_file, "r", encoding='utf-8-sig') as infile:
            lines = infile.readlines()
            lines[0] = lines[0].lstrip("\ufeff")

        for i, line in enumerate(lines):
            line = line.split(",")
            label = line[-1].rstrip("\n")

            # arrange coordinates into 2d array form [ [x1, y1] ... [x4, y4] ]
            size = 4
            coordinates = line[:-1]
            single_coordinates = [[] for _ in range(size)]
            for j in range(0, 2 * size, 2):
                single_coordinates[j // 2] = [coordinates[j], coordinates[j + 1]]
            quad_coordinates.append(np.array(single_coordinates, dtype=int))

            # Remove entries with ### since they are invalid
            if label == "###" or label == "*":
                bool_tags.append(True)
            else:
                # remove non-alphanumeric characters
                pattern = re.compile(r"[^\w\d\s]")
                label = re.sub(pattern, "", label)

                # if transcription is empty after removing non-alphanumerics, skip
                if label == "":
                    continue

                bool_tags.append(False)

            text_tags.append(label)

        return quad_coordinates, text_tags, np.array(bool_tags, dtype=np.bool)


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


if __name__ == "__main__":
    imagePath = "./Dataset/Train/TrainImages/"
    annotationPath = "./Dataset/Train/TrainTruth/"

    # example usage
    dataset = TextLocalizationSet(imagePath, annotationPath, (126, 224))
    sample = dataset.__getitem__(0)
    image = Image.open(sample['path'])
    box = detect(sample['score'], sample['geometry'])
    plot_img = plot_boxes(image, box)
    plot_img.show()
    # print(dataset[0])

    # can use DataLoader now with this custom dataset
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate)
