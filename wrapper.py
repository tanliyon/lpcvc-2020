import sys
import os
from sampler.iframes_no_mv import sampler as iFRAMES
from ctc.ctc import ctc_recognition as CTC
from test import OCR_att
from detector.inference import detection as DETECTOR
from PIL import Image
import numpy as np
import time

def main(argv):
    
    if len(argv) != 3:
        raise ValueError('Incorrect number of input arguements.\nExpected: Video path, Question file, Answer File')
    
    video_path = argv[0]
    question_path = argv[1]
    answer_path = argv[2]

    # Open question text file and create a list of questions
    try:
        with open(question_path, 'r') as question_file:
            for line in question_file:
                questions = line.split(";")
    except:
        raise ValueError("Error obtaining questions from input file")
    questions = questions[:-1]
    
    # get relevant frames from video
    # list of frames
    start = time.time()
    frames_list = iFRAMES(video_path)
    if len(frames_list) == 0:
        raise ValueError('No frames received')
    interval = time.time() - start
    print("Sampling Block took %d minutes %.3f seconds" % (interval//60, interval%60))

    # get bounding box coordinates for all frames
    start = time.time()
    frames_list, bboxes = DETECTOR('./frames')
    interval = time.time() - start
    print("Detection Block took %d minutes %.3f seconds" % (interval//60, interval%60))
    
    # get list of recognised strings from frames
    start = time.time()
    text_list = OCR_att(frames_list, bboxes)
    if len(text_list) == 0:
        raise ValueError('No text recognised')
    interval = time.time() - start
    print("Recognition Block took %d minutes %.3f seconds" % (interval//60, interval%60))

    # Answers the questions
    # text_list format: [[string, string, string], [string, string, string, string],....]
    text_dict = dict()   
    for words_in_frame in text_list:
        if [words for words in words_in_frame if words in questions]:
            for word in words:
                if word in text_dict:
                    text_dict[word] = text_dict[word] + [x for x in words_in_frame if x != word]
                else:
                    text_dict[word] = [x for x in words if x != word]
    
    for question in questions:
        if question not in text_dict.keys():
            text_dict[question] = '-'

    # Write answers to questions from text_dict dictionary                
    try:
        with open(answer_path, "w") as answer_file:
            for question in text_dict.keys():
                answer_file.write(question + ": ")
                answer_file.writelines(text_dict[question]) #is space between words included here
                answer_file.write("; ")

    except:
        raise ValueError('Error writing answers to output file')

if __name__ == "__main__":
    main(sys.argv[1:])
