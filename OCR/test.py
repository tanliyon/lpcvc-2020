from data_manager.data_manager import textDataset, syn_text
from utils.config import CONFIG
import torchvision.transforms as transforms
from utils.util import *
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import time
from torch.utils.tensorboard import SummaryWriter
import models.crnn_quant as crnn
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from itertools import compress

image = torch.FloatTensor(4, 3, 32, 32)
SOS_token = 0
EOS_TOKEN = 1
BLANK = 2
to_pil = transforms.ToPILImage()


def OCR_att(frames, bboxes):
    max_length = 16
    converter = strLabelConverterForAttention(CONFIG.ALPHABETS, ":")
    encoder = crnn.CNN(32,1,256)
    number_classes = len(CONFIG.ALPHABETS.split(":")) + 3
    decoder = crnn.decoderV2(256, number_classes, dropout_p=0.1)
    encoder.load_state_dict(torch.load("/home/damini/PycharmProjects/low_power_attention_model/lpcvc-2020/OCR/OUTPUT_SYNTH3_DATA/encoder_2.pth"))
    decoder.load_state_dict(torch.load("/home/damini/PycharmProjects/low_power_attention_model/lpcvc-2020/OCR/OUTPUT_SYNTH3_DATA/decoder_2.pth"))
    to_bw = transforms.Compose([transforms.Resize((32, 280), interpolation=Image.NEAREST), transforms.ToTensor()])
    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False
    encoder.eval()
    decoder.eval()
    word_list = []
    prob = [1] * len(bboxes[0])
    for p, (frame, frame_bboxes) in enumerate(zip(frames, bboxes)):
        words = []
        frame = to_pil(frame)
        for box in frame_bboxes:
            tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = box
            top = min(tl_y, tr_y)
            left = min(tl_x, bl_x)
            height = max(bl_y, br_y) - min(tl_y, tr_y)
            width = max(tr_x, br_x) - min(tl_x, bl_x)
            img = transforms.functional.crop(frame, top, left, height, width)
            words.append(to_bw(transforms.functional.crop(frame, top, left, height, width)))

        words = torch.stack(words)
        decoded_word = list()
        encoder_outputs = encoder(words)
        decoder_input = torch.zeros(words.shape[0]).long()
        decoder_input = decoder_input
        decoder_hidden = decoder.initHidden(words.shape[0])
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            probs = torch.exp(decoder_output)
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            temp = [converter.decode(item) for item in ni]
            prob = [p * float(probs[i, item]) for i, (item, p) in enumerate(zip(ni, prob))]
            decoded_word.append(temp)

        decoded_word = list(map(list, zip(*decoded_word)))
        decoded_word = [word for i, word in enumerate(decoded_word) if prob[i] > 0.6]
        decoded_word = ["".join(item).replace("$", "")  for item in decoded_word]
        word_list.append(decoded_word)
    return word_list


if __name__ == '__main__':
    frame = Image.open("trial/img_54.jpg")
    bw = transforms.Grayscale()
    t1 = transforms.ToTensor()
    frame = t1(bw(frame))
    # OCR_att(frame, [[[573,97,625,106,625,125,572,116],[542,166,627,179,627,204,542,191], [251,227,359,237,359,259,251,249], [635,182,675,189,674,218,635,211]]])
    OCR_att(frame, [[[542,166,627,179,627,204,542,191], [251,227,359,237,359,259,251,249], [520,500,500,537,559,559,500,500]]])
    # OCR_att(frame, [[[542,166,627,179,627,204,542,191]]])





