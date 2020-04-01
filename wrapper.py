import sys

from PIL import Image
import numpy as np
import time

from sampler.iframes_no_mv import sampler as iFRAMES
from ctc.ctc import ctc_recognition as CTC
from detector.inference import detection as DETECTOR


def main():
    
    if len(sys.argv) != 4:
        raise ValueError('Incorrect number of input arguements.\nExpected: Video path, Question file, Answer File')
    
    video_path = sys.argv[1]
    question_path = sys.argv[2]
    answer_path = sys.argv[3]

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
    #frames_list = iFRAMES(video_path)
    #if len(frames_list) == 0:
    #    raise ValueError('No frames received')
    interval = time.time() - start
    print("Sampling Block took %d minutes %.3f seconds" % (interval//60, interval%60))

    # get bounding box coordinates for all frames
    start = time.time()
    frames_list, bboxes = DETECTOR('./frames')
    interval = time.time() - start
    print("Detection Block took %d minutes %.3f seconds" % (interval//60, interval%60))
    
    # get list of recognised strings from frames
    start = time.time()
    text_list = CTC(frames_list, bboxes)
    interval = time.time() - start
    print("Recognition Block took %d minutes %.3f seconds" % (interval//60, interval%60))

    # Answers the questions
    # text_list format: [[string, string, string], [string, string, string, string],....]
    ans_dict = {question: "" for question in questions}

    """ Uncomment if want to print predicted words from all frames
    for text in text_list:
        print(text)
    """

    # Brute force way to go through all the questions and write corresponding answers
    # (Can be improved I believe but for now)
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
