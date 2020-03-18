# Implemented based on: https://github.com/BelBES/crnn-pytorch

import os
import string
import torch
import math
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# from WordRecognitionSet import collate
# from WordRecognitionSet import WordRecognitionSet
from data_manager import syn_text
from model import CRNN

parser = argparse.ArgumentParser(description='Train the CTC model.')
parser.add_argument('--epoch', default=1,
                    type=int, help='Number of epochs to train the model')
parser.add_argument('--learning_rate', default=0.01,
                    type=float, help='Base learning rate to start training with')
parser.add_argument('--weight_decay', default=0.001,
                    type=float, help='Weight decay to train the model on')
parser.add_argument('--save_dir', default='trained_model',
                    type=str, help='Name of directory to save trained model')
parser.add_argument('--load_model', default=None,
                    type=str, help='Relative path to load trained model')
parser.add_argument('--batch_size', default=4,
                    type=int, help='Number of images to train in one batch')
parser.add_argument('--print_iter', default=500,
                    type=int, help='Number of iterations before printing info')
parser.add_argument('--model_name', default="model",
                    type=str, help='Name of model')
parser.add_argument('--verbose', default=True,
                    type=bool, help='Tooggle to print info while tranining')
parser.add_argument('-debug', action='store_true')
args = parser.parse_args()

PIX_P_PRED = 30  # Width of pixels per prediction
RESIZE_W = 600
RESIZE_H = 50

def string_to_index(labels):
	index = []
	length = []

	for label in labels:
		prev_char = ""
		cur_length = 0

		for char in label:
			if char >= '0' and char <= '9':
				index.append(ord(char)-ord('0') + 1)
			elif char >= 'a' and char <='z':
				index.append(ord(char)-ord('a') + 11)
			else:
				index.append(ord(char)-ord('A') + 37)

			cur_length += 1
			prev_char = char

		length.append(cur_length)

	return torch.IntTensor(index), torch.IntTensor(length)

def val_loss(net, loader):
    net.eval()
    running_loss = 0

    for i, data in enumerate(loader):
        img, labels = data

        img = img[0]
        labels_ind = string_to_index(labels)
        target_length = torch.IntTensor([len(labels_ind)])

        while (img.size[0] < (len(labels_ind)*PIX_P_PRED)):
        	img = transforms.functional.resize(img, (int(img.size[1]*1.1), int(img.size[0]*1.1)))

        img = to_tensor(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        preds = net(img)
        input_length = torch.IntTensor([preds.shape[0]])

        loss = criterion(preds.cpu(), labels_ind.cpu(), input_length.cpu(), target_length.cpu())
        running_loss += loss

    return running_loss/i


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((RESIZE_H, RESIZE_W)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    net = CRNN()
    net.to(device)
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    if args.load_model:
        load_path = os.path.join(os.getcwd(), args.load_model)
        net.load_state_dict(torch.load(load_path))

    save_dir = os.path.join(os.getcwd(), args.save_dir)
    if not os.path.isdir(save_dir):
    	os.mkdir(save_dir)

    train_data = syn_text("annotation_train.txt", "E:\\cam2\\lpcvc-2020\\ctc\\mnt\\ramdisk\\max\\90kDICT32px", transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                             shuffle=True, num_workers=1
                             )

    val_data = syn_text("annotation_val.txt", "E:\\cam2\\lpcvc-2020\\ctc\\mnt\\ramdisk\\max\\90kDICT32px", transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                             shuffle=True, num_workers=1
                             )

    running_loss=0

    for i in range(args.epoch):
	    for j, data in enumerate(train_loader):
		    # Zero the parameter gradients
	        optimizer.zero_grad()

	        img, labels = data
	        labels_ind, target_length = string_to_index(labels)

	        img = img.to(device)
	        preds = net(img)
	        input_length = torch.IntTensor(preds.shape[1])
	        input_length = input_length.fill_(preds.shape[0])

	        loss = criterion(preds.cpu(), labels_ind.cpu(), input_length.cpu(), target_length.cpu())
	        loss.backward()

	        print(loss)
	        print(labels)

	        nn.utils.clip_grad_norm(net.parameters(), 10.0) #Clip gradient temp
	        optimizer.step()

	        if args.debug:
	        	print(net.decode_seq(preds))
	        	print(labels)
	        	print(loss)
	        	print(labels_ind)
	        	if j == 10:
	        		break

		    # print statistics
	        running_loss += loss
	        if j % args.print_iter == args.print_iter-1 and args.verbose:
	            print('epoch: %d (%d) | loss: %.3f' %
	                (i+1, j+1, running_loss/args.print_iter))
	            print(net.decode_seq(preds))
	            print(labels)
	            running_loss = 0
	            torch.save(net.state_dict(), os.path.join(args.save_dir, args.model_name + ".pth"))
	            break

	        lr_scheduler.step()

	    if args.debug:
        	break