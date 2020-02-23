# Implemented based on: https://github.com/BelBES/crnn-pytorch
import os
import string
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from matplotlib.path import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from WordRecognitionSet import collate
from WordRecognitionSet import WordRecognitionSet
from model import CRNN

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	#TODO: Handle input of varying sizes. For now the height needs to be 32
    net = CRNN()
    net.to(device)

    cap = cv2.VideoCapture(0)

	if cap.isOpened():
		cv2.namedWindow("Word Recognize", cv2.WINDOW_AUTOSIZE)