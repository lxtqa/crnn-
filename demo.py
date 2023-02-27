import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import sys

import models.crnn as crnn

def load_model():
    model_path = './data/crnn.pth'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

    model = crnn.CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))

    converter = utils.strLabelConverter(alphabet)

    transformer = dataset.resizeNormalize((100, 32))
    return transformer, model, converter

def demo(img_path,transformer, model, converter):

    image = Image.open(img_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)


    confi, preds = preds.max(2)

    confi = confi.data

    confi = confi.transpose(1, 0).contiguous().view(-1)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    confidence = 0
    char_num = 0

    bconfidence = 0
    blank_num = 0


    for i in range(len(raw_pred)):
        if raw_pred[i] != "-":
            char_num = char_num + 1
            confidence = confidence + confi[i].item()
        else:
            blank_num = blank_num + 1
            bconfidence = bconfidence + confi[i].item()
    if char_num == 0:
        confidence = - sys.maxsize - 1
    else:
        confidence = confidence/char_num

    if blank_num == 0:
        bconfidence = - sys.maxsize - 1
    else:
        bconfidence = bconfidence/blank_num
    return confidence, bconfidence, raw_pred, sim_pred

if __name__ == "__main__":
    
    confi, raw_pred, sim_pred = demo('./data/demo.jpg')
    print("confidence = " + str(confi))
    print('%-20s => %-20s' % (raw_pred, sim_pred))