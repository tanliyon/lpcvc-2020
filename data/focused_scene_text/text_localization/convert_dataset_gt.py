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


def convert_dataset_train_gt():
    gt_dir = sys.argv[1]
    gt_file_names = sorted_alphanumeric(os.listdir(gt_dir))

    for file_name in gt_file_names:
        file_path = os.path.join(gt_dir, file_name)

        with open(file_path, "r") as file_handle:
            gt_lines = file_handle.readlines()

        # overwrite the content of same file
        file_handle = open(file_path, "w")

        for line in gt_lines:
            one_gt = line.split(" ")

            transcription = one_gt[-1].lstrip("\"")
            transcription = transcription.rstrip("\"\n")

            separator = ","
            converted_one_gt = separator.join([one_gt[0], one_gt[1],
                                               one_gt[2], one_gt[1],
                                               one_gt[2], one_gt[3],
                                               one_gt[0], one_gt[3],
                                               transcription])
            file_handle.write(converted_one_gt + "\n")


def convert_dataset_test_gt():
    gt_dir = sys.argv[1]
    gt_file_names = sorted_alphanumeric(os.listdir(gt_dir))

    for file_name in gt_file_names:
        file_path = os.path.join(gt_dir, file_name)

        with open(file_path, "r") as file_handle:
            gt_lines = file_handle.readlines()

        # overwrite the content of same file
        file_handle = open(file_path, "w")

        for line in gt_lines:
            one_gt = line.split(", ")

            transcription = one_gt[-1].lstrip("\"")
            transcription = transcription.rstrip("\"\n")

            separator = ","
            converted_one_gt = separator.join([one_gt[0], one_gt[1],
                                               one_gt[2], one_gt[1],
                                               one_gt[2], one_gt[3],
                                               one_gt[0], one_gt[3],
                                               transcription])
            file_handle.write(converted_one_gt + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "train_gt":
        convert_dataset_train_gt()
    elif sys.argv[1] == "test_gt":
        convert_dataset_test_gt()
