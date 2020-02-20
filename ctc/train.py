# Implemented based on: https://github.com/BelBES/crnn-pytorch

import os
import string
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
# from transforms.functional import resize

from WordRecognitionSet import collate
from WordRecognitionSet import WordRecognitionSet
from model import CRNN

parser = argparse.ArgumentParser(description='Train the CTC model.')
parser.add_argument('--epoch', default=1,
                    type=int, help='Number of epochs to train the model')
parser.add_argument('--learning_rate', default=0.001,
                    type=int, help='Base learning rate to start training with')
parser.add_argument('--weight_decay', default=0.001,
                    type=int, help='Weight decay to train the model on')
parser.add_argument('--save_dir', default='trained_model',
                    type=str, help='Name of directory to save trained model')
parser.add_argument('--load_model', default=None,
                    type=str, help='Relative path to load trained model')
parser.add_argument('--batch_size', default=1,
                    type=int, help='Number of images to train in one batch')
parser.add_argument('--print_iter', default=100,
                    type=int, help='Number of iterations before printing info')
parser.add_argument('--verbose', default=True,
                    type=bool, help='Tooggle to print info while tranining')
parser.add_argument('-debug', action='store_true')
args = parser.parse_args()


# def string_to_index(labels):
# 	index = []
# 	max_length = 0

# 	for label in labels:
# 		out = []
# 		for char in label:
# 			if char >= '0' and char <= '9':
# 				out.append(ord(char)-ord('0'))
# 			elif char >= 'a' and char <='z':
# 				out.append(ord(char)-ord('a') + 10)
# 			else:
# 				out.append(ord(char)-ord('A') + 36)
# 		if len(out) > max_length:
# 			max_length = len(out)
# 		index.append(out)

# 	for l in index:
# 		while len(l) < max_length:
# 			l.append(-1)

# 	return torch.IntTensor(index)

def string_to_index(labels):
	index = []

	for label in labels:
		for char in label:
			if char >= '0' and char <= '9':
				index.append(ord(char)-ord('0') + 1)
			elif char >= 'a' and char <='z':
				index.append(ord(char)-ord('a') + 11)
			else:
				index.append(ord(char)-ord('A') + 37)
	return torch.IntTensor(index)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	#TODO: Handle input of varying sizes. For now the height needs to be 32
    net = CRNN()
    net.to(device)
    criterion = nn.CTCLoss(zero_infinity=True)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.load_model:
        load_path = os.path.join(os.getcwd(), args.load_model)

	# TODO: Create directory to save_path if it does not exist
	# TODO: Check for valid path
    save_dir = os.path.join(os.getcwd(), args.save_dir)
    if not os.path.isfile(save_dir):
    	os.mkdir(save_dir)

    if args.load_model:
        net.load_state_dict(torch.load(args.load_path))

    train_data = WordRecognitionSet(train=True,
                                 transform=None)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                             shuffle=True, num_workers=1,
                             collate_fn=collate)
    # resize = transforms.Resize(32)
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    running_loss=0

    for i in range(args.epoch):
	    for j, data in enumerate(train_loader):
		    # Zero the parameter gradients
	        optimizer.zero_grad()

	        img, _, labels = data

	        # plt.imshow(img[0].permute(1,2,0))
	        # plt.show()

	        # img = torch.stack(img)
	        img = img[0]

	        # if (img.size[1] < 32):
	        # 	img = transforms.functional.resize(img, (int(img.size[1]*1.5), int(img.size[0]*1.5)))
	        while (img.size[0] < len(labels[0])*30):
	        	img = transforms.functional.resize(img, (int(img.size[1]*1.1), int(img.size[0]*1.1)))

	        img = to_tensor(img)
	        img = img.unsqueeze(0)
	        img = img.to(device)

	        preds = net(img)
	        input_length = torch.IntTensor([preds.shape[0]])
	        target_length = torch.IntTensor([len(label) for label in labels])
	        labels_ind = string_to_index(labels)

	        try:
	        	loss = criterion(preds, labels_ind.cpu(), input_length.cpu(), target_length.cpu())
	        except:
	        	loss = criterion(preds, labels_ind.cuda(), input_length.cuda(), target_length.cuda())
	        # loss = criterion(preds, labels_ind.cuda(), input_length.cuda(), target_length.cuda())
	        loss.backward()

	        nn.utils.clip_grad_norm(net.parameters(), 10.0) #Clip gradient temp
	        optimizer.step()

	        if args.debug:
	        	print(net.decode_seq(preds))
	        	print(labels)
	        	print(loss)
	        	if j == 100:
	        		break

		    # print statistics
	        running_loss += loss
	        if j % args.print_iter == args.print_iter-1 and args.verbose:
	            print('epoch: %d (%d) | loss: %.3f' %
	                (i+1, j+1, running_loss/args.print_iter))
	            running_loss = 0
	            print(net.decode_seq(preds))
	            print(labels)

        # Save model
	    torch.save(net.state_dict(), os.path.join(args.save_dir, str(i+1) + ".pth"))


# Notes:
# Number of prediction = width of conv output
# 1 prediction = 30 pixel wide