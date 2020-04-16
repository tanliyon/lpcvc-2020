import os
import re
import torch
import numpy as np
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import *
from detect import *
from error import *

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
    path = [item["path"] for item in batch]
    image = [item["image"] for item in batch]
    score = [item["score"] for item in batch]
    geometry = [item["geometry"] for item in batch]
    mask = [item["mask"] for item in batch]
    return [path, image, score, mask, geometry]


class TextLocalizationSet(Dataset):
    def __init__(self, image_directory_path, annotation_directory_path, new_dimensions):
        self.image_directory_path = image_directory_path
        self.annotation_directory_path = annotation_directory_path
        self.image_paths, self.image_names, self.annotation_paths = self.Get_Images_Path_Name_Annotation()
        self.new_dimensions = new_dimensions
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.image_directory_path, self.image_paths[idx])
        image = self.Load_Image(image_path)
        quad_coordinates, bool_tags = self.Load_Annotation(os.path.join(self.annotation_directory_path, self.annotation_paths[idx]))

        image, quad_coordinates = self.Resize_Data(image, quad_coordinates)

        score_map, training_mask, geometry_map = Load_Geometry_Score_Maps(quad_coordinates, bool_tags, self.new_dimensions[0], self.new_dimensions[1])

        sample = {
            "path": image_path,
            "image": self.transform(image),
            "score": score_map,
            "geometry": geometry_map,
            "mask": training_mask
        }

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

        :return:      image (PIL Image): image data
        """
        image = Image.open(image_path)
        image = image.convert("L")
        check_number_of_channels(image)

        return image

    def Load_Annotation(self, annotation_file):
        """
        :description: Loads the ground truth coordinates of the text box and text labels from annotation file of image

        :param:       image_name (str): name of the image in question

        :return:      quad_coordinates (numpy array(4, 8)): 8 coordinates of ground truth box
                      text_tags (bool numpy array): ground truth text labels
        """
        quad_coordinates = []
        bool_tags = []

        with open(annotation_file, "r", encoding='utf-8-sig') as infile:
            lines = infile.readlines()

        for i, line in enumerate(lines):
            quad_coordinates.append(list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
            line = line.split(",")
            label = line[-1].rstrip("\n")
            # Remove entries with ### since they are invalid
            if label == "###" or label == "*":
                bool_tags.append(True)
            else:
                # remove non-alphanumeric characters
                pattern = re.compile(r"[^\w\d\s]")
                label = re.sub(pattern, "", label)

                # if transcription is empty after removing non-alphanumerics, skip
                if label == "":
                    bool_tags.append(True)
                    continue

                bool_tags.append(False)

        return np.array(quad_coordinates), np.array(bool_tags, dtype=np.bool) #,text_tags

    def Resize_Data(self, image, quad_coordinates):
        old_height, old_width = image.height, image.width
        new_height, new_width = self.new_dimensions
        ratio_height, ratio_width = new_height / old_height, new_width / old_width

        resized_image = image.resize((new_width, new_height), Image.BILINEAR)

        resized_coordinates = np.zeros(quad_coordinates.shape)
        if quad_coordinates.size > 0:
            resized_coordinates[:, [0, 2, 4, 6]] = quad_coordinates[:, [0, 2, 4, 6]] * ratio_width
            resized_coordinates[:, [1, 3, 5, 7]] = quad_coordinates[:, [1, 3, 5, 7]] * ratio_height

        return resized_image, resized_coordinates

if __name__ == "__main__":
    imagePath = "./Dataset/Train/TrainImages/"
    annotationPath = "./Dataset/Train/TrainTruth/"

    # example usage
    dataset = TextLocalizationSet(imagePath, annotationPath, (126, 224))
    sample = dataset.__getitem__(0)
    image = Image.open(sample['path'])
    box = get_boxes(sample['score'].cpu().numpy(), sample['geometry'].cpu().numpy())
    # box = get_boxes(sample['score'], sample['geometry'])
    plot_img = plot_boxes(sample['image'], box)
    plot_img.show()

    # can use DataLoader now with this custom dataset
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate)
