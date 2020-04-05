import sys
import numpy as np
import torchvision
import torch
import time
import threading
from PIL import Image
from mpipe import UnorderedStage, Stage, Pipeline
from sampler.iframes_no_mv import sampler as iFRAMES
from ctc.ctc import CTC as CTC
from detector.inference import Detector as DETECTOR
from torchvision import transforms

def main():
    if len(sys.argv) != 4:
        raise ValueError('Incorrect number of input arguements.\nExpected: Video path, Question file, Answer File')
    
    video_path = sys.argv[1]
    question_path = sys.argv[2]
    answer_path = sys.argv[3]
    transform = transforms.Compose([
                transforms.Resize((126, 224)),
		transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
            ])

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
    print("Sampling took %d minutes %.3f seconds" % (interval//60, interval%60))

    # Create pipeline
    start = time.time()
    detector = Stage(DETECTOR, 1)
    recognition = Stage(CTC, 1)
    detector.link(recognition)
    pipe = Pipeline(detector)
    text_list = []

    # Execute pipeline
    frames_data = torchvision.datasets.ImageFolder(root='./frames', transform=transform)
    frames_loader = torch.utils.data.DataLoader(frames_data, batch_size=1, shuffle=False)

    for i, frame in enumerate(frames_loader):
        pipe.put(frame[0])
        break

    # Wait until threads are completed
    for i in range(len(frames_data)):
        text_list.append(pipe.get())

    # Stop pipeline
    pipe.put(None)
    
    interval = time.time() - start
    print("Detection & Recogntion took %d minutes %.3f seconds" % (interval//60, interval%60))

    # Answers the questions
    # text_list format: [[string, string, string], [string, string, string, string],....]
    ans_dict = {question: "" for question in questions}

    # Uncomment if want to print predicted words from all frames
    #for text in text_list:
    #    print(text)

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

