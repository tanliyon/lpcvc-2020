import sys
import os
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


def convert_file_names():
    """
    function to convert names of the files (images or ground truth)
    in order to merge with existing set of data
    :return: No return
    """
    img_folder_dir = sys.argv[1]

    img_dir_list = sorted_alphanumeric(os.listdir(img_folder_dir))

    label = int(sys.argv[2])
    for img in img_dir_list:
        # for Mac OS
        if img == ".DS_Store":
            continue
        img_dir = os.path.join(img_folder_dir, img)
        new_dir = os.path.join(img_folder_dir, f"img_{label}.jpg")  # manually change here
        os.rename(img_dir, new_dir)
        label += 1


if __name__ == "__main__":
    convert_file_names()
