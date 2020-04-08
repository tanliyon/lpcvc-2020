from torchvision import transforms
from OCR.utils.config import CONFIG
from OCR.utils.util import *
import OCR.models.crnn_lang_chenjun as crnn
import numpy as np


def crop(img, box, transform):
    tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = box
    top = min(tl_y, tr_y)
    left = min(tl_x, bl_x)
    height = max(bl_y, br_y) - min(tl_y, tr_y)
    width = max(tr_x, br_x) - min(tl_x, bl_x)
    return transform(transforms.functional.crop(img, top, left, height, width))

def init_attn():
    encoder = crnn.CNN(32,1,256)
    number_classes = len(CONFIG.ALPHABETS.split(":")) + 3
    decoder = crnn.decoderV2(256, number_classes, dropout_p=0.1)
    encoder.load_state_dict(torch.load("./encoder.pth", map_location='cpu'))
    decoder.load_state_dict(torch.load("./decoder.pth", map_location='cpu'))
    return encoder, decoder

def attn_OCR(encoder, decoder, img, boxes):
    attn_transform = transforms.Compose([
        transforms.Resize((32, 280)),
        transforms.ToTensor()
    ])
    
    EOS_TOKEN = 1
    max_length = 15
    to_pil = transforms.ToPILImage()
    converter = strLabelConverterForAttention(CONFIG.ALPHABETS, ":")
    pil_img = []
    
    for im in img:
        pil_img.append(to_pil(im))
    
    words = []
    count = []
    
    for i, box in enumerate(boxes):
        if type(box).__module__ != np.__name__ or len(box) == 0:
                continue
        count.append(len(box))
        for b in box:
            words.append(crop(pil_img[i], b, attn_transform))
    
    words = torch.stack(words)
    word_list = []
    prob = [1] * sum(count)
    
    decoded_word = list()
    with torch.no_grad():
        encoder_outputs = encoder(words)
    decoder_input = torch.zeros(words.shape[0]).long()
    decoder_hidden = decoder.initHidden(words.shape[0])
    
    for di in range(max_length):
        with torch.no_grad():
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
    decoded_word = ["".join(item).replace("$", "") for item in decoded_word]
    word_list.append(decoded_word)
    
    return word_list, count




