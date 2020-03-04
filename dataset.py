import os
import csv
import glob
import logging
import numpy as np
import torch.utils.data as data
from error import *

class Dataset(data.Dataset):
    def __init__(self, imagePath, annotationPath):
        self.imagePath = imagePath
        self.annotationPath = annotationPath

    def loadImages(self, mode="training"):
        does_directory_exist(self.imagePath)

        imagePaths = []
        imageNames = []

        for extension in ["jpg", "png", "jpeg", "JPG"]:
            imagePaths.extend(glob.glob(os.path.join(self.imagePath, "*.{}".format(extension))))

        for i in range(0, len(imagePaths)):
            imageNames.append(imagePaths[i].split("/")[-1])

        logging.info("\nPreparing {} images for {}\n".format(len(imagePaths), mode))
        return sorted(imagePaths), sorted(imageNames)

    def loadAnnotation(self, annotationFile):
        annotationFile = os.path.join(self.annotationPath, "gt_" + annotationFile + ".txt")
        does_file_exist(annotationFile)

        quad_coordinates = []
        text_tags = []

        with open(annotationFile, "r") as annotation:
            reader = csv.reader(annotation)
            for line in reader:
                line = [i.strip("\ufeff").strip("\xef\xbb\xbf") for i in line]
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:8]))
                quad_coordinates.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                text_tags.append(True) if line[-1] == "*" or line[-1] == "###" else text_tags.append(False)

        return np.array(quad_coordinates, dtype=np.float32), np.array(text_tags, dtype=np.bool)

if __name__ == '__main__':
    imagePath = "./Dataset/Train/TrainImages/"
    annotationPath = "./Dataset/Train/TrainTruth/"
    dataset = Dataset(imagePath, annotationPath)

    imagePaths, imageNames = dataset.loadImages()
    dataset.loadAnnotation("img_1")
