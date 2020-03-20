from data_manager.data_manager import textDataset, syn_text
from utils.config import CONFIG
# from tool_util import *
import torch.nn as nn
import torchvision.transforms as transforms
import torch
device = torch.device("cuda:0")
from utils.util import *
from torch.utils.data import Dataset, DataLoader
import random
import torch.optim as optim
import torch.utils.data
import time
from torch.utils.tensorboard import SummaryWriter
import models.crnn_lang_chenjun as crnn
from models.inception_test import CNN as test_model
import matplotlib.pyplot as plt

image = torch.cuda.FloatTensor(4, 3, 32, 32)
image = torch.cuda.FloatTensor(4, 3, 32, 32)
SOS_token = 0
EOS_TOKEN = 1
BLANK = 2

def test(encoder, decoder, batchsize, data_loader):
    print('Start testing...')

    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False

    encoder.eval()
    decoder.eval()
    val_iter = iter(data_loader)

    n_correct = 0
    n_total = 0
    loss_avg = Averager()
    max_iter = 1000
    max_iter = min(max_iter, len(data_loader))
    # max_iter = len(data_loader) - 1
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        b = cpu_images.size(0)
        with torch.no_grad():
            image.resize_(cpu_images.size()).copy_(cpu_images)

        target_variable, target_variable_length = converter.encode(cpu_texts)
        n_total += len(cpu_texts[0]) + 1
        decoded_words = []
        decoded_label = []
        decoder_attentions = torch.zeros(len(cpu_texts[0]) + 1, 71)
        encoder_outputs = encoder(image)
        target_variable = target_variable.cuda()
        decoder_input = target_variable[0].cuda()
        decoder_hidden = decoder.initHidden(b).cuda()
        loss = 0.0
        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di-1] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            if ni == EOS_TOKEN:
                decoded_words.append('<EOS>')
                decoded_label.append(EOS_TOKEN)
                break
            else:
                decoded_words.append(converter.decode(ni))
                decoded_label.append(ni)


        for pred, target in zip(decoded_label, target_variable[1:,:]):
            if pred == target:
                n_correct += 1

        if i % 50 == 0:
            texts = cpu_texts[0]
            print('pred:%-20s, gt: %-20s' % (decoded_words, texts))


    accuracy = n_correct / float(n_total)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    print('Accuracy: %f' % (accuracy))


if __name__ == "__main__":
    #load model
    # model = Model()
    #load data
    t0 = time.time()
    fix_width  = True
    # transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
    transform = transforms.Compose([transforms.Resize((32, 280), interpolation=Image.NEAREST), transforms.ToTensor()])
    # test_dataset = syn_text("annotation_test.txt", "/home/damini/Downloads/mnt/ramdisk/max/90kDICT32px/",transform)
    test_dataset = textDataset("gt_new.txt", "/home/damini/Documents/icdar_2015/train","data", "train", transform)
    # test_loader = DataLoader(test_dataset, batch_size= 1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size= 1, shuffle=True)
    writer = SummaryWriter()
    # criterion = torch.nn.NLLLoss()
    converter = strLabelConverterForAttention(CONFIG.ALPHABETS, ":")
    # encoder = test_model(32, 1, 530)
    encoder = crnn.CNN(32,3,256)
    number_classes = len(CONFIG.ALPHABETS.split(":")) + 3
    decoder = crnn.decoderV2(256, number_classes, dropout_p=0.1)
    encoder.load_state_dict(torch.load("/home/damini/PycharmProjects/low_power_attention_model/OCR/OUTPUT_SYNTH_DATA/encoder_2.pth"))
    decoder.load_state_dict(torch.load("/home/damini/PycharmProjects/low_power_attention_model/OCR/OUTPUT_SYNTH_DATA/decoder_2.pth"))
    encoder.cuda()
    decoder.cuda()
    t1 = time.time()

    loss_avg = Averager()
    for epoch in range(100):
        model_paths = "OUTPUT_SYNTH2_DATA"
        test(encoder, decoder, 1, test_loader)
        # writer.close()