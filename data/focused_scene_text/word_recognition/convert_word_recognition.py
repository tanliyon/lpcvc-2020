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
    img_folder_dir = sys.argv[2]

    img_dir_list = sorted_alphanumeric(os.listdir(img_folder_dir))

    label = int(sys.argv[3])
    for img in img_dir_list:
        if img == ".DS_Store":
            continue
        img_dir = os.path.join(img_folder_dir, img)
        new_dir = os.path.join(img_folder_dir, f"word_{label}.png")
        os.rename(img_dir, new_dir)
        label += 1


def convert_content():
    gt_file_dir = sys.argv[2]

    with open(gt_file_dir, "r") as infile:
        all_gt = infile.readlines()

    pattern = re.compile(r"(?P<file_name>word_(\d)+.png)")

    label = int(sys.argv[3])
    for idx in range(len(all_gt)):
        all_gt[idx] = re.sub(pattern, f"word_{label}.png", all_gt[idx])
        label += 1

    with open(gt_file_dir, "w") as outfile:
        for gt in all_gt:
            outfile.write(gt)


if __name__ == "__main__":
    """
    sys.argv[1]: specify mode
    """
    mode = sys.argv[1]

    if mode == "-f":
        convert_file_names()

    elif mode == "-c":
        convert_content()
