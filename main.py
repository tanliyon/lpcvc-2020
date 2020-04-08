import sys
from PIL import Image
from torchvision import transforms
import torchvision
import torch
import numpy as np
import time

from sampler.iframes import sampler as iFRAMES
from detector.inference import init_EAST, detect
from OCR.test import init_attn, attn_OCR
from ctc.ctc import init_CTC, CTC_OCR
from detector.detect import plot_boxes, adjust_ratio
from PIL import Image

USE_ATTN_OCR = True
SHOW_BOXES = True
SHOW_TEXT = True
BATCH_SIZE = 32

# Helper Functions
def list_2_2dlist(lst, count):
    idx = 0
    for c in count:
        yield(lst[idx:idx+c])
        idx += c
        
def show_boxes(index, dataset, boxes):
    ind = 0
    for i in range(index*BATCH_SIZE, (index+1)*BATCH_SIZE):
        if i >= len(dataset):
            break
        img, _ = dataset[i]
        boxes[ind] = adjust_ratio(boxes[ind], 1/5.7, 1/5.7)
        box_img = plot_boxes(img, boxes[ind])
        box_img.save('./detected_imgs/img{}.bmp'.format(i))
        ind += 1
        
def main():
    if len(sys.argv) != 3:
        raise ValueError('Incorrect number of input arguements.\nExpected: Video path, Question file')
    
    video_path = sys.argv[1]
    question_path = sys.argv[2]
    answer_path = 'answers.txt'

    # Open question text file and create a list of questions
    try:
        with open(question_path, 'r') as question_file:
            for line in question_file:
                questions = line.split(";")
    except:
        raise ValueError("Error obtaining questions from input file")
    questions = questions[:-1]

    # get relevant frames from video
    start = time.time()
    # frames_list = iFRAMES(video_path)
    interval = time.time() - start
    print("Sampling Block took %d minutes %.3f seconds" % (interval//60, interval%60))
    
    # List transforms
    detector_transform = transforms.Compose([
        transforms.Resize((126, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    # Init models
    detector = init_EAST()
    if USE_ATTN_OCR:
        encoder, decoder = init_attn()
    else:
        OCR = init_CTC()
    
    # Initlialize dataset and dataloader
    ori_img = torchvision.datasets.ImageFolder(root='./frames', transform=None)
    frames_data = torchvision.datasets.ImageFolder(root='./frames', transform=detector_transform)
    frames_loader = torch.utils.data.DataLoader(frames_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Loop through dataloader
    text_list = []
    to_pil = transforms.ToPILImage(mode="L")

    for i, data in enumerate(frames_loader):
        img, _ = data
        boxes = detect(detector, img)
        predictions, count = attn_OCR(encoder, decoder, img, boxes) if USE_ATTN_OCR else CTC_OCR(OCR, img, boxes)
        
        if SHOW_BOXES:
            show_boxes(i, ori_img, boxes)
        
        for lst in list(list_2_2dlist(predictions, count)):
            text_list.append(lst)

    # Answers the questions
    ans_dict = {question: "" for question in questions}

    # Uncomment if want to print predicted words from all frames
    if SHOW_TEXT:
        for text in text_list:
            print(text)

    # Brute force way to go through all the questions and write corresponding answers
    for question in questions:
        for words_in_frame in text_list:
            if question in words_in_frame:
                for word in words_in_frame:
                    if word != question:
                        ans_dict[question] += " " + word

        if ans_dict[question] == "":
            ans_dict[question] = " -"

    # Write answers to questions from ans_dict dictionary
    try:
        with open(answer_path, "w") as answer_file:
            for question in ans_dict.keys():
                answer_file.write(question + ":")
                answer_file.writelines(ans_dict[question]) #is space between words included here: Yes
                answer_file.write(";")

    except:
        raise ValueError('Error writing answers to output file')


if __name__ == "__main__":
    main()
