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

def val(encoder, decoder, criterion, batchsize, data_loader, teach_forcing=False, max_iter=100):
    print('Start val')

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
        if not teach_forcing:

            for di in range(1, target_variable.shape[0]):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                loss_avg.add(loss)
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
    writer.add_scalar('Loss/test', loss_avg.val(), epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(train_iter, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, teach_forcing_prob=1):

    data = train_iter.next()
    cpu_images, cpu_texts = data
    if(len([text for text in cpu_texts if text == ""]) != 0):
        return torch.tensor(0.0)
    b = cpu_images.size(0)
    target_variable, target_variable_length = converter.encode(cpu_texts)
    with torch.no_grad():
        image.resize_(cpu_images.size()).copy_(cpu_images)

    encoder_outputs = encoder(image)
    target_variable = target_variable.cuda()
    decoder_input = target_variable[0].cuda()
    decoder_hidden = decoder.initHidden(b).cuda()
    loss = 0.0
    for di in range(1, target_variable.shape[0]):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_variable[di])
        topv, topi = decoder_output.data.topk(1)
        ni = topi.squeeze()
        decoder_input = ni
    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss




#TODO:code doesnt run with 1 image for trianing - fix it

if __name__ == "__main__":
    #load model
    # model = Model()
    #load data
    t0 = time.time()
    fix_width  = True
    # transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((32, 280), interpolation=Image.NEAREST), transforms.ToTensor()])
    dataset = textDataset("gt_new.txt", "/home/damini/Documents/icdar_2015/train","data", "train", transform)
    dataset = syn_text("annotation_train.txt", "/home/damini/Downloads/mnt/ramdisk/max/90kDICT32px/",transform)
    # val_dataset = textDataset("gt_new.txt", "/home/damini/Documents/icdar_2015/val","data", "val", transform)
    val_dataset = syn_text("annotation_val.txt", "/home/damini/Downloads/mnt/ramdisk/max/90kDICT32px/",transform)
    # sampler = randomSequentialSampler(dataset, CONFIG.BATCH_SIZE)
    train_loader = DataLoader(dataset, batch_size= CONFIG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= 1, shuffle=True)
    writer = SummaryWriter()
    criterion = torch.nn.CrossEntropyLoss()
    converter = strLabelConverterForAttention(CONFIG.ALPHABETS, ":")
    encoder = crnn.CNN(32,1,256)
    number_classes = len(CONFIG.ALPHABETS.split(":")) + 3
    decoder = crnn.decoderV2(256, number_classes, dropout_p=0.1)
    encoder.apply(weights_init)
    decoder.apply(weights_init)
    encoder.cuda()
    decoder.cuda()

    # writer.add_graph(decoder)
    # writer.close()
    # decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0005, betas=(0.5, 0.999))
    # encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0005, betas=(0.5, 0.999))

    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001, betas=(0.5, 0.999), eps=1e-08, weight_decay=0.00004)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001, betas=(0.5, 0.999), eps=1e-08, weight_decay=0.00004)
    t1 = time.time()

    loss_avg = Averager()
    for epoch in range(100):
        train_iter = iter(train_loader)
        min_iter = 10000
        i = 0
        while i < 10000:
            for e, d in zip(encoder.parameters(), decoder.parameters()):
                e.requires_grad = True
                d.requires_grad = True
            encoder.train()
            decoder.train()
            cost = trainBatch(train_iter, encoder, decoder, criterion, encoder_optimizer,
                              decoder_optimizer, teach_forcing_prob=0.5)
            loss_avg.add(cost)
            i += 1

            if i % 5000 == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, 300, i, len(train_loader), loss_avg.val()), end=' ')
                loss_avg.reset()
                t1 = time.time()
                print('time elapsed s%d' % (t1 - t0))
                t0 = time.time()
            # break
        model_paths = "OUTPUT_SYNTH3_DATA"
        val(encoder, decoder, criterion, 1, val_loader, teach_forcing=False)
        torch.save(
            encoder.state_dict(), '{0}/encoder_{1}.pth'.format(model_paths, epoch))
        torch.save(
            decoder.state_dict(), '{0}/decoder_{1}.pth'.format(model_paths, epoch))

        writer.add_scalar('Loss/train', loss_avg.val(), epoch)
        # writer.close()

