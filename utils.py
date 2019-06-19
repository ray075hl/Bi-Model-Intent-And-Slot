import torch
import numpy as np


def make_mask(real_len, max_len=50, label_size=123, batch=16):
    mask = torch.zeros(batch, max_len, label_size)
    for index, item in enumerate(real_len):
        mask[index, :item, :] = 1.0
    return mask


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)

        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def one_hot(array, Num=123, maxlen=50):

    shape = array.size()
    batch = shape[0]
    if len(shape) == 1:
        res = torch.zeros(batch, Num)
        for i in range(batch):
            res[i][array[i]] = 1
    else:
        res = torch.zeros(batch, maxlen, Num)
        for i in range(batch):
            for j in range(maxlen):
                res[i][j][array[i, j]] = 1

    return res

import random

def get_batch(data, batch_size=16):
    total_num = len(data)
    i = 0
    while i < total_num//batch_size:
        sentence = []
        real_len = []
        slot_label = []
        intent_label = []
        if batch_size == 1:
            sentence.append(data[i][0])
            real_len.append(data[i][1])
            slot_label.append(data[i][2])
            intent_label.append(data[i][3])
        else:
            index = np.random.choice(total_num, batch_size)
            # print('index: ', index)
            for m in index:

                sentence.append(data[m][0])
                real_len.append(data[m][1])
                slot_label.append(data[m][2])
                intent_label.append(data[m][3])

        i += 1
        yield (sentence, real_len, slot_label, intent_label)

def getBatch(data, batch_size):
    random.shuffle(data)
    sindex = 0
    eindex = batch_size
    while eindex < len(data):

        sentence = []
        real_len = []
        slot_label = []
        intent_label = []
         
        batch = data[sindex:eindex]
        for m in range(sindex, eindex):
            sentence.append(data[m][0])
            real_len.append(data[m][1])
            slot_label.append(data[m][2])
            intent_label.append(data[m][3])

        temp = eindex
        eindex = eindex + batch_size
        sindex = temp

        yield batch

