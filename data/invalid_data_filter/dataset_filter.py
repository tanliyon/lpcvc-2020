"""
    usage: python dataset_filter.py path_to_image_folder path_to_gt_folder
"""
import os
import sys

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


if __name__ == "__main__":
    """
    argv[1]: path to image folder
    argv[2]: path to ground truth folder
    """
    # feed in path to image folder
    img_dir = sys.argv[1]
    # feed in path to ground truth folder
    gt_dir = sys.argv[2]

    img_dir_list = sorted_alphanumeric(os.listdir(img_dir))
    gt_dir_list = sorted_alphanumeric(os.listdir(gt_dir))

    file_removed = 0
    for file_index, file in enumerate(gt_dir_list):
        file_dir = os.path.join(gt_dir, file)

        with open(file_dir, "r", encoding="utf-8-sig") as gt_file:
            gt_content = gt_file.readlines()

        for index in range(len(gt_content)):
            gt_content[index] = gt_content[index].rstrip("\n")
            gt_content[index] = gt_content[index].split(",")

        count = 0
        for gt in gt_content:
            if gt[-1] != "###":
                count += 1

        if count == 0:
            os.remove(file_dir)
            os.remove(os.path.join(img_dir, img_dir_list[file_index]))
            file_removed += 1

    print(f"number of files removed: {file_removed}")

    # renaming the files in order
    img_dir_list = sorted_alphanumeric(os.listdir(img_dir))
    gt_dir_list = sorted_alphanumeric(os.listdir(gt_dir))
    for index in range(len(img_dir_list)):
        src_path = os.path.join(img_dir, img_dir_list[index])
        dest_path = os.path.join(img_dir, f"img_{index+1}.jpg")
        os.rename(src_path, dest_path)

        src_path = os.path.join(gt_dir, gt_dir_list[index])
        dest_path = os.path.join(gt_dir, f"gt_img_{index+1}.txt")
        os.rename(src_path, dest_path)