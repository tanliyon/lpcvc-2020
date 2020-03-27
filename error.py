import os
import sys
import glob

def does_directory_exist(path):
    if not os.path.isdir(path):
        raise OSError("Directory path {} provided does not exist".format(path))

def does_file_exist(path):
    if not os.path.isfile(path):
        raise OSError("File path {} provided does not exist".format(path))

def is_valid_text_file(path):
    for file in path:
        if file.lower().endswith(".txt"):
            continue
        else:
            raise TypeError("File path {} provided is not a valid annotation file".format(file))

def is_valid_image_file(path):
    for file in path:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        else:
            raise TypeError("File path {} provided is not a valid image file".format(file))

def is_empty_coordinates(coordinates):
    if coordinates.shape[0] == 0:
        return True

def is_valid_ratio(ratios):
    for ratio in ratios:
        if ratio < 1:
            raise ValueError("Ratio {} cannot be less than 1".format(ratio))

def check_number_of_channels(image):
    channels = len(image.getbands())
    if channels != 1:
        raise TypeError("Image must have only 1 channel instead of {} channels".format(channels))
