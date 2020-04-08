import math
import numpy as np
from PIL import Image
from cv2 import fillPoly
import torch
import torchvision
from torch import nn
from dataset import *
from error import *
from mpl_toolkits.mplot3d import Axes3D

def Get_Boundary_Values(coordinates):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinates
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

def Calculate_Distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def Calculate_Area(coordinates):
    x_min, x_max, y_min, y_max = Get_Boundary_Values(coordinates)
    area = (x_max - x_min) * (y_max - y_min)
    return area

def Move_Points(coordinates, offset1, offset2, reference_length, coefficient):
    x1_index = (offset1 % 4) * 2 + 0
    y1_index = (offset1 % 4) * 2 + 1
    x2_index = (offset2 % 4) * 2 + 0
    y2_index = (offset2 % 4) * 2 + 1

    r1 = reference_length[offset1 % 4]
    r2 = reference_length[offset2 % 4]

    x_length = coordinates[x1_index] - coordinates[x2_index]
    y_length = coordinates[y1_index] - coordinates[y2_index]
    length = Calculate_Distance(coordinates[x1_index], coordinates[y1_index], coordinates[x2_index], coordinates[y2_index])

    if length > 1:
    	ratio = (r1 * coefficient) / length
    	coordinates[x1_index] += ratio * (-x_length)
    	coordinates[y1_index] += ratio * (-y_length)

    	ratio = (r2 * coefficient) / length
    	coordinates[x2_index] += ratio * x_length
    	coordinates[y2_index] += ratio * y_length

    return coordinates

def Shrink_Coordinates(coordinates, coefficient=0.3):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinates

    r1 = min(Calculate_Distance(x1, y1, x2, y2), Calculate_Distance(x1, y1, x4, y4))
    r2 = min(Calculate_Distance(x2, y2, x1, y1), Calculate_Distance(x2, y2, x3, y3))
    r3 = min(Calculate_Distance(x3, y3, x2, y2), Calculate_Distance(x3, y3, x4, y4))
    r4 = min(Calculate_Distance(x4, y4, x1, y1), Calculate_Distance(x4, y4, x3, y3))
    reference_length = [r1, r2, r3, r4]

	# obtain offset to perform move_points() automatically
    if(Calculate_Distance(x1, y1, x2, y2) + Calculate_Distance(x3, y3, x4, y4)) > (Calculate_Distance(x2, y2, x3, y3) + Calculate_Distance(x1, y1, x4, y4)):
	    offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
	    offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = coordinates.copy()
    v = Move_Points(v, 0 + offset, 1 + offset, reference_length, coefficient)
    v = Move_Points(v, 2 + offset, 3 + offset, reference_length, coefficient)
    v = Move_Points(v, 1 + offset, 2 + offset, reference_length, coefficient)
    v = Move_Points(v, 3 + offset, 4 + offset, reference_length, coefficient)

    #return np.reshape(coordinates, (4, 2))
    return v

def Rotate_Coordinates(coordinates, angle, anchor=None):
    v = coordinates.reshape((4, 2)).T

    if anchor is None:
        anchor = v[:, :1]

    rotated_angle = Get_Rotation_Matrix(angle)
    rotated_coordinate = np.dot(rotated_angle, v - anchor)
    return (rotated_coordinate + anchor).T.reshape(-1)

def Calculate_Angle_Error(coordinates):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinates
    x_min, x_max, y_min, y_max = Get_Boundary_Values(coordinates)

    error = Calculate_Distance(x1, y1, x_min, y_min) + Calculate_Distance(x2, y2, x_max, y_max) + Calculate_Distance(x3, y3, x_min, y_min) + Calculate_Distance(x4, y4, x_max, y_max)
    return error

# def _Validate_Coordinates(self, quad_coordinates, text_tags):
#     if is_empty_coordinates(quad_coordinates):
#         return quad_coordinates, text_tags
#
#     quad_coordinates[:, :, 0] = np.clip(quad_coordinates[:, :, 0], 0,  self.width - 1)
#     quad_coordinates[:, :, 1] = np.clip(quad_coordinates[:, :, 1], 0, self.height - 1)
#
#     validated_coordinates = []
#     validated_tags = []
#
#     for coordinate, tag in zip(quad_coordinates, text_tags):
#         area = self._Calculate_Area(coordinate)
#
#         if abs(area) < 1:
#             #logging.info("\nInvalid coordinates {} with area {} for image {}\n".format(coordinate, area, self.image_name))
#             continue
#         if area > 0:
#             #logging.info("\nCoordinates {} in wrong direction with area {} for image {}\n".format(coordinate, area, self.image_name))
#             coordinate = coordinate[(0, 3, 2, 1), :]
#         validated_coordinates.append(coordinate)
#         validated_tags.append(tag)
#
#     return np.array(validated_coordinates), np.array(validated_tags)

def Get_Rotation_Angle(coordinates):
    angle_interval = 1
    angles = list(range(-90, 90, angle_interval))
    areas = []

    for angle in angles:
        rotated_coordinate = Rotate_Coordinates(coordinates, angle / 180 * math.pi)
        areas.append(Calculate_Area(rotated_coordinate))

    sorted_area_index = sorted(list(range(len(areas))), key=lambda k : areas[k])
    minimum_error = math.inf
    best_index = -1
    rank = 10

    for i in sorted_area_index[:rank]:
        rotated_coordinate = Rotate_Coordinates(coordinates, angles[i] / 180 * math.pi)
        error = Calculate_Angle_Error(rotated_coordinate)

        if error < minimum_error:
            minimum_error = error
            best_index = i

    return angles[best_index] / 180 * math.pi

def Get_Rotation_Matrix(angle):
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

def Rotate_Pixels(rotated_angle, height, width, anchor_x, anchor_y):
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, y.size))

    coordinate_map = np.concatenate((x_lin, y_lin), 0)
    rotated_coordinates = np.dot(rotated_angle, coordinate_map - np.array([[anchor_x], [anchor_y]])) + np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coordinates[0, :].reshape(x.shape)
    rotated_y = rotated_coordinates[1, :].reshape(y.shape)

    return rotated_x, rotated_y

def Resize_Height(image, quad_coordinates, ratio=0.2):
    ratio = 1 + ratio * (2 * np.random.rand() - 1)
    old_height = image.height
    new_height = int(np.around(old_height * ratio))

    resized_image = image.resize((image.width, new_height), Image.BILINEAR)

    resized_coordinates = quad_coordinates.copy()
    if quad_coordinates.size > 0:
        resized_coordinates[:, [1, 3, 5, 7]] = quad_coordinates[:, [1, 3, 5, 7]] * (new_height / old_height)

    return resized_image, resized_coordinates

def Rotate_Data(image, quad_coordinates, range=10):
    mid_x = ( image.width - 1) / 2
    mid_y = (image.height - 1) / 2

    angle = range * (2 * np.random.rand() - 1)

    rotated_image = image.rotate(angle, Image.BILINEAR)

    rotated_coordinates = np.zeros(quad_coordinates.shape)
    for i, coordinate in enumerate(quad_coordinates):
        rotated_coordinates[i, :] = self.Rotate_Coordinates(coordinate, -angle / 180 * math.pi, np.array([[mid_x],[mid_y]]))

    return rotated_image, rotated_coordinates

def Load_Geometry_Score_Maps(quad_coordinates, text_tags, height, width, scale=0.25):
    score_map     = np.zeros((int(math.ceil(height * scale)), int(width * scale), 1), dtype=np.float32)
    training_mask = np.zeros((int(math.ceil(height * scale)), int(width * scale), 1), dtype=np.float32)  # mask used during traning, to ignore some hard areas
    geometry_map  = np.zeros((int(math.ceil(height * scale)), int(width * scale), 5), dtype=np.float32)

    x = np.arange(0,  width, int(1 / scale))
    y = np.arange(0, height, int(1 / scale))
    index_x, index_y = np.meshgrid(x, y)

    polys = []
    ignored_polys = []

    for i, item in enumerate(zip(quad_coordinates, text_tags)):
        coordinate = item[0]
        tag = item[1]

        if tag:
            ignored_polys.append(np.around(scale * coordinate.reshape((4, 2))).astype(np.int32))
            continue

        #shrink_coordinates = np.around(self._Shrink_Coordinates(scale * coordinate.copy().flatten())).astype(np.int32)
        shrink_coordinates = np.around(scale * Shrink_Coordinates(coordinate).reshape((4, 2))).astype(np.int32)
        polys.append(shrink_coordinates)

        poly_mask = np.zeros(score_map.shape[:-1], np.float32)
        fillPoly(poly_mask, [shrink_coordinates], 1)

        angle = Get_Rotation_Angle(coordinate)
        rotated_angle = Get_Rotation_Matrix(angle)

        rotated_coordinates = Rotate_Coordinates(coordinate, angle)
        x_min, x_max, y_min, y_max = Get_Boundary_Values(rotated_coordinates)
        rotated_x, rotated_y = Rotate_Pixels(rotated_angle, height, width, coordinate[0], coordinate[1])

        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0

        geometry_map[:, :, 0] += d1[index_y, index_x] * poly_mask
        geometry_map[:, :, 1] += d2[index_y, index_x] * poly_mask
        geometry_map[:, :, 2] += d3[index_y, index_x] * poly_mask
        geometry_map[:, :, 3] += d4[index_y, index_x] * poly_mask
        geometry_map[:, :, 4] += angle * poly_mask

    fillPoly(score_map, polys, 1)
    fillPoly(training_mask, ignored_polys, 1)
    # print("True Score {}".format(score_map))
    # print("True Geometry {}".format(geometry_map))
    # cv2.imshow("Score", score_map)
    # cv2.waitKey(0)

    return torch.Tensor(score_map).permute(2, 0, 1), torch.Tensor(training_mask).permute(2, 0, 1), torch.Tensor(geometry_map).permute(2, 0, 1)

"""
Testing and sample usage of functions
"""
if __name__ == '__main__':
    imagePath = "./Dataset/Train/TrainImages/"
    annotationPath = "./Dataset/Train/TrainTruth/"
    dataset = Utils(imagePath, annotationPath)
    dataset.__getitem__(0)

    # imagePaths, imageNames = dataset.getImagesPathName()
    # image = dataset.loadImage(imagePaths[0])
    # dataset.loadAnnotation("img_1")
