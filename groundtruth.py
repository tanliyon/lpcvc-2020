import os
import csv
import glob
import math
import logging
import numpy as np
from PIL import Image
import cv2
from cv2 import fillPoly
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import torchvision
from torch import nn
from dataset import *
from error import *
from mpl_toolkits.mplot3d import Axes3D

class GroundTruthGeneration:
    def __init__(self, image_name, image):
        self.image_name = image_name
        self.image = image
        self.height, self.width, _ = image.shape

    def _Calculate_Area(self, quad_coordinates):
        edge = [(quad_coordinates[1][0] - quad_coordinates[0][0]) * (quad_coordinates[1][1] + quad_coordinates[0][1]),
                (quad_coordinates[2][0] - quad_coordinates[1][0]) * (quad_coordinates[2][1] + quad_coordinates[1][1]),
                (quad_coordinates[3][0] - quad_coordinates[2][0]) * (quad_coordinates[3][1] + quad_coordinates[2][1]),
                (quad_coordinates[0][0] - quad_coordinates[3][0]) * (quad_coordinates[0][1] + quad_coordinates[3][1])]

        return float(np.sum(edge) / 2)

    def _Calculate_Distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _Move_Points(self, coordinates, offset1, offset2, reference_length, coefficient):
        x1_index = (offset1 % 4) * 2 + 0
        y1_index = (offset1 % 4) * 2 + 1
        x2_index = (offset2 % 4) * 2 + 0
        y2_index = (offset2 % 4) * 2 + 1

        r1 = reference_length[offset1 % 4]
        r2 = reference_length[offset2 % 4]

        x_length = coordinates[x1_index] - coordinates[x2_index]
        y_length = coordinates[y1_index] - coordinates[y2_index]
        length = self._Calculate_Distance(coordinates[x1_index], coordinates[y1_index], coordinates[x2_index], coordinates[y2_index])

        if length > 1:
        	ratio = (r1 * coefficient) / length
        	coordinates[x1_index] += ratio * (-x_length)
        	coordinates[y1_index] += ratio * (-y_length)

        	ratio = (r2 * coefficient) / length
        	coordinates[x2_index] += ratio * x_length
        	coordinates[y2_index] += ratio * y_length

        return coordinates

    def _Shrink_Coordinates(self, coordinates, coefficient=0.3):
        # reference_length = [None, None, None, None]
        # for j in range(4):
        #     reference_length[j] = min(np.linalg.norm(coordinates[j] - coordinates[(j + 1) % 4]), np.linalg.norm(coordinates[j] - coordinates[(j - 1) % 4]))

        x1, y1, x2, y2, x3, y3, x4, y4 = coordinates

        r1 = min(self._Calculate_Distance(x1, y1, x2, y2), self._Calculate_Distance(x1, y1, x4, y4))
        r2 = min(self._Calculate_Distance(x2, y2, x1, y1), self._Calculate_Distance(x2, y2, x3, y3))
        r3 = min(self._Calculate_Distance(x3, y3, x2, y2), self._Calculate_Distance(x3, y3, x4, y4))
        r4 = min(self._Calculate_Distance(x4, y4, x1, y1), self._Calculate_Distance(x4, y4, x3, y3))
        reference_length = [r1, r2, r3, r4]

    	# obtain offset to perform move_points() automatically
        if(self._Calculate_Distance(x1, y1, x2, y2) + self._Calculate_Distance(x3, y3, x4, y4)) > (self._Calculate_Distance(x2, y2, x3, y3) + self._Calculate_Distance(x1, y1, x4, y4)):
    	    offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
        else:
    	    offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

        coordinates = self._Move_Points(coordinates, 0 + offset, 1 + offset, reference_length, coefficient)
        coordinates = self._Move_Points(coordinates, 2 + offset, 3 + offset, reference_length, coefficient)
        coordinates = self._Move_Points(coordinates, 1 + offset, 2 + offset, reference_length, coefficient)
        coordinates = self._Move_Points(coordinates, 3 + offset, 4 + offset, reference_length, coefficient)

        return np.reshape(coordinates, (4, 2))

    def _Validate_Coordinates(self, quad_coordinates, text_tags):
        if is_empty_coordinates(quad_coordinates):
            return quad_coordinates, text_tags

        quad_coordinates[:, :, 0] = np.clip(quad_coordinates[:, :, 0], 0,  self.width - 1)
        quad_coordinates[:, :, 1] = np.clip(quad_coordinates[:, :, 1], 0, self.height - 1)

        validated_coordinates = []
        validated_tags = []

        for coordinate, tag in zip(quad_coordinates, text_tags):
            area = self._Calculate_Area(coordinate)

            if abs(area) < 1:
                logging.info("\nInvalid coordinates {} with area {} for image {}\n".format(coordinate, area, self.image_name))
                continue
            if area > 0:
                logging.info("\nCoordinates {} in wrong direction with area {} for image {}\n".format(coordinate, area, self.image_name))
                coordinate = coordinate[(0, 3, 2, 1), :]
            validated_coordinates.append(coordinate)
            validated_tags.append(tag)

        return np.array(validated_coordinates), np.array(validated_tags)

    def Get_Score_Geometry_Map(self, quad_coordinates, text_tags, scale):
        score_map = np.zeros((int(self.height * scale), int(self.width * scale), 1), dtype=np.float32)
        poly_mask = np.zeros((int(self.height * scale), int(self.width * scale), 1), dtype=np.uint8)
        training_mask = np.ones((int(self.height * scale), int(self.width * scale), 1), dtype=np.uint8)  # mask used during traning, to ignore some hard areas
        geometry_map  = np.zeros((int(self.height * scale), int(self.width * scale), 8), dtype=np.float32)

        polys = []
        ignored_polys = []

        for i, item in enumerate(zip(quad_coordinates, text_tags)):
            coordinate = item[0]
            tag = item[1]

            if tag:
                ignored_polys.append(np.around(scale * coordinate).astype(np.int32))
                continue

            #shrink_coordinates = self._Shrink_Coordinates(coordinate.copy().flatten(), reference_length).astype(np.int32)[np.newaxis, :, :]
            shrink_coordinates = np.around(self._Shrink_Coordinates(scale * coordinate.copy().flatten())).astype(np.int32)
            polys.append(shrink_coordinates)

            poly_mask = np.zeros(score_map.shape[:-1], np.float32)
            fillPoly(poly_mask, [shrink_coordinates], 1)

            geometry_map[:, :, 0] += coordinate[0][0] * poly_mask
            geometry_map[:, :, 1] += coordinate[0][1] * poly_mask
            geometry_map[:, :, 2] += coordinate[1][0] * poly_mask
            geometry_map[:, :, 3] += coordinate[1][1] * poly_mask
            geometry_map[:, :, 4] += coordinate[2][0] * poly_mask
            geometry_map[:, :, 5] += coordinate[2][1] * poly_mask
            geometry_map[:, :, 6] += coordinate[3][0] * poly_mask
            geometry_map[:, :, 7] += coordinate[3][1] * poly_mask

            #fillPoly(poly_mask, shrink_coordinates, i + 1)

            # if the poly is too small, then ignore it during training
            # poly_h = min(np.linalg.norm(coordinate[0] - coordinate[3]), np.linalg.norm(coordinate[1] - coordinate[2]))
            # poly_w = min(np.linalg.norm(coordinate[0] - coordinate[1]), np.linalg.norm(coordinate[2] - coordinate[3]))
            # if min(poly_h, poly_w) < FLAGS.min_text_size:
            #     fillPoly(training_mask, coordinate.astype(np.int32)[np.newaxis, :, :], 0)
            # if tag:
            #     fillPoly(training_mask, coordinate.astype(np.int32)[np.newaxis, :, :], 0)
        #
        # print(self.image_name)
        # cv2.imshow("Score", score_map)
        # cv2.waitKey(0)
        fillPoly(score_map, polys, 1)
        fillPoly(training_mask, ignored_polys, 1)

        #return np.moveaxis(score_map, 2, 0), np.moveaxis(training_mask, 2, 0), np.moveaxis(geometry_map, 2, 0)
        return torch.Tensor(score_map).permute(2, 0, 1), torch.Tensor(training_mask).permute(2, 0, 1), torch.Tensor(geometry_map).permute(2, 0, 1)

    def Load_Geometry_Score_Maps(self, quad_coordinates, text_tags, scale=0.25):
        quad_coordinates, text_tags = self._Validate_Coordinates(quad_coordinates, text_tags)
        return self.Get_Score_Geometry_Map(quad_coordinates, text_tags, scale)

# class Dataset(data.Dataset):
#     """
#     :description: Custom dataset class
#
#     :attr:        image_directory_path (str): path to image directory in train or test directory
#                   annotation_directory_path (str): path to ground truth annotation directory in train or test directory
#                   image_paths (list): list of all images in image directory
#                   image_names (list): list of all image names in image directory
#     """
#     def __init__(self, image_directory_path, annotation_directory_path):
#         """
#         :param:       image_directory_path (str): path to image directory in train or test directory
#                       annotation_directory_path (str): path to ground truth annotation directory in train or test directory
#         """
#         self.image_directory_path = image_directory_path
#         self.annotation_directory_path = annotation_directory_path
#         self.image_paths, self.image_names = self.Get_Images_Path_Name()
#
#
#     def __getitem__(self, i):
#         """
#         :description: Gets the ground truth score map and geometry map of the image in question
#
#         :param:       i (int): index of image in question in the list of image paths
#
#         :return:      image (uint8 numpy array): image data (channel, height, width)
#                       score_map (): ground truth score map of image
#                       geometry_map (): ground truth geometry map of image
#                       training_mask ():
#         """
#
#         image = self.Load_Image(self.image_paths[i])
#         quad_coordinates, text_tags = self.Load_Annotation(self.image_names[i])
#         gt = GroundTruthGeneration(self.image_names[i], image)
#         score_map, geometry_map, training_mask = gt.Load_Geometry_Score_Maps(quad_coordinates, text_tags)
#         return image, score_map, geometry_map, training_mask
#
#
#     def __len__(self):
#         """
#         :description: Gets number of images in train or test directory
#
#         :return:      (int): number of images in directory
#         """
#         return len(self.image_names)
#
#
#     def Get_Images_Path_Name(self, mode="training"):
#         """
#         :description: Gets all image paths and image names from train or test directory
#
#         :return:      image_paths (list): sorted image paths in directory
#                       image_names (list): sorted image names in directory
#         """
#         does_directory_exist(self.image_directory_path)
#
#         image_paths = []
#         image_names = []
#
#         for extension in ["jpg", "png", "jpeg", "JPG"]:
#             image_paths.extend(glob.glob(os.path.join(self.image_directory_path, "*.{}".format(extension))))
#
#         for i in range(0, len(image_paths)):
#             image_names.append(image_paths[i].split("/")[-1])
#
#         logging.info("\nPreparing {} images for {}\n".format(len(image_paths), mode))
#         return sorted(image_paths), sorted(image_names)
#
#
#     def Load_Image(self, image_path):
#         """
#         :description: Loads the image from image file and makes it a numpy array
#
#         :param:       image_path (str): path of the image file in question
#
#         :return:      image (uint8 numpy array): image data (channel, height, width)
#         """
#         image = np.array(Image.open(image_path))    # type -> uint8
#         return np.moveaxis(image, 2, 0)             # reverses order -> channel x height x width
#
#
#     def Load_Annotation(self, image_name):
#         """
#         :description: Loads the ground truth coordinates of the text box and text labels from annotation file of image
#
#         :param:       image_name (str): name of the image in question
#
#         :return:      quad_coordinates (float32 numpy array): 8 coordinates of ground truth box
#                       text_tags (bool numpy array): ground truth text labels
#         """
#         annotation_file = os.path.join(self.annotation_directory_path, "gt_" + image_name.split(".")[0] + ".txt")
#         does_file_exist(annotation_file)
#
#         quad_coordinates = []
#         text_tags = []
#
#         with open(annotation_file, "r") as annotation:
#             reader = csv.reader(annotation)
#             for line in reader:
#                 line = [i.strip("\ufeff").strip("\xef\xbb\xbf") for i in line]
#                 x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:8]))
#                 quad_coordinates.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
#                 text_tags.append(True) if line[-1] == "*" or line[-1] == "###" else text_tags.append(False)
#
#         return np.array(quad_coordinates, dtype=np.float32), np.array(text_tags, dtype=np.bool)


"""
Testing and sample usage of functions
"""
if __name__ == '__main__':
    imagePath = "./Dataset/Train/TrainImages/"
    annotationPath = "./Dataset/Train/TrainTruth/"
    dataset = Dataset(imagePath, annotationPath)
    dataset.__getitem__(0)

    # imagePaths, imageNames = dataset.getImagesPathName()
    # image = dataset.loadImage(imagePaths[0])
    # dataset.loadAnnotation("img_1")
