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

from TextLocalizationSet import TextLocalizationSet
from model import CRNN

parser = argparse.ArgumentParser(description='Train the CTC model.')
parser.add_argument('--epoch', default=50,
                    type=int, help='Number of epochs to train the model')
parser.add_argument('--learning_rate', default=0.01,
                    type=int, help='Base learning rate to start training with')
parser.add_argument('--weight_decay', default=0.001,
                    type=int, help='Base learning rate to start training with')
parser.add_argument('--save_dir', default='trained_model',
                    type=str, help='Name of directory to save trained model')
parser.add_argument('--load_model', default=None,
                    type=str, help='Relative path to load trained model')
parser.add_argument('--batch_size', default=4,
                    type=int, help='Number of images to train in one batch')
parser.add_argument('--print_iter', default=100,
                    type=int, help='Number of iterations before printing info')
parser.add_argument('--verbose', default=True,
                    type=bool, help='Tooggle to print info while tranining')
parser.add_argument('-debug', action='store_true')
args = parser.parse_args()

def show_image(img, coords, transcriptions):
    # img = np.swapaxes(img, 0, 2)
    # img = img.permute(1, 2, 0)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for coor in coords:
        coor = list(coor)
        coor.append(coor[0])
        codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,]

        path = Path(coor, codes)
        rect = patches.PathPatch(path, lw=0.5, color='red', fill=False)
        ax.add_patch(rect)

    print(transcriptions)
    plt.show()
    return

def crop_words(img, coords):
	words = []
	to_pil = transforms.ToPILImage()
	to_tensor = transforms.ToTensor()
	img = to_pil(img)

	for coor in coords:
		top_left, top_right, bot_right, bot_left = coor
		left_x = min(bot_left[0], top_left[0])
		right_x = max(bot_right[0], top_right[0])
		top_y = min(top_left[1], top_right[1])
		bot_y = max(bot_left[1], bot_right[1])
		# TODO: Handle case when the box is vertical
		# TODO: Handle empty list
		words.append(to_tensor(transforms.functional.resized_crop(img, top_y, left_x, bot_y-top_y, right_x-left_x, (32, 300), interpolation=2)))

	return torch.stack(words)

def collate(batch):
    data = [item[0] for item in batch]
    coords = [item[1] for item in batch]
    transcriptions = [item[2] for item in batch]
    return [data, coords, transcriptions]

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	#TODO: Handle input of varying sizes. For now the height needs to be 32
	net = CRNN()
	net.to(device)
	criterion = nn.CTCLoss(reduction="None")
	optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

	if args.load_model:
	    load_path = os.path.join(os.getcwd(), args.load_model)

	# TODO: Create directory to save_path if it does not exist
	# TODO: Check for valid path
	save_dir = os.path.join(os.getcwd(), args.save_dir)
	if args.load_model:
	    net.load_state_dict(torch.load(args.load_path))

	train_data = TextLocalizationSet(img_dir="train", gt_dir="train_gt")
	train_loader = DataLoader(train_data, batch_size=args.batch_size,
                          	  shuffle=True, num_workers=1, collate_fn=collate)
	test_data = TextLocalizationSet(img_dir="test", gt_dir="test_gt")

	for i, data in enumerate(train_loader):
	    imgs, coords, transcriptions = data
	    words = []

	    if args.debug:
	    	show_image(imgs[0], coords[0], transcriptions[0])

	    for i, img in enumerate(imgs):
	    	if i == 0:
	    		words = crop_words(img, coords[i])
	    	else:
	    		words = torch.cat((words, crop_words(img, coords[i])))

	    if args.debug:
	    	one_word = words[0].permute(1, 2, 0)
	    	plt.imshow(one_word)
	    	plt.show()
	    	print(words.shape)
	    	net.decode = True

	    words = words.to(device)
	    # zero the parameter gradients
	    optimizer.zero_grad()

	    # forward + backward + optimize
	    preds = net(words)
	    print(preds)
	    loss = criterion(outputs, labels)
	    loss.backward()
	    optimizer.step()

	    # print statistics
	    running_loss += loss.item()
	    if i % args.print_iter == args.print_iter-1 and args.verbose:
	        print('[%d, %5d] loss: %.3f' %
	            (epoch + 1, i + 1, running_loss / 2000))
	        running_loss = 0.0

	        # Save model
	        torch.save(model.state_dict(), args.save_dir)




# Notes:
# Number of prediction = width of conv output
