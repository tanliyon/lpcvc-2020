import re
import string
translator = str.maketrans('', '', string.punctuation)
with open("/home/damini/Documents/icdar_2015/val/gt.txt") as file:
    with open("/home/damini/Documents/icdar_2015/val/gt_new.txt", "w") as new_file:
        lines = file.readlines()
        for idx in range(len(lines)):
            line = re.findall(r'(word_[\d]*.png), \"(.+?)\"', lines[idx])
            gt = line[0][1]
            new_gt = gt.translate(translator)
            new_gt = line[0][0] + ", \"" + new_gt + "\"\n"
            new_file.write(new_gt)
            print("done")


