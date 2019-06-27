import torch
import numpy as np
from config import max_len, batch
from make_dict import slot_dict


def make_mask(real_len, max_len=max_len, label_size=len(slot_dict), batch=batch):
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


def one_hot(array, Num=len(slot_dict), maxlen=max_len):

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
                if array[i, j] == Num:
                    pass
                else:
                    res[i][j][array[i, j]] = 1

    return res

import random

def get_batch(data, batch_size=batch):
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

        yield (sentence, real_len, slot_label, intent_label)

def get_chunks(labels):
    chunks = []
    start_idx,end_idx = 0,0
    for idx in range(1,len(labels)-1):
        chunkStart, chunkEnd = False,False
        if labels[idx-1] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            prevTag, prevType = labels[idx-1][:1], labels[idx-1][2:]
        else:
            prevTag, prevType = 'O', 'O'
        if labels[idx] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            Tag, Type = labels[idx][:1], labels[idx][2:]
        else:
            Tag, Type = 'O', 'O'
        if labels[idx+1] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            nextTag, nextType = labels[idx+1][:1], labels[idx+1][2:]
        else:
            nextTag, nextType = 'O', 'O'

        if (Tag == 'B' and prevTag in ('B', 'I', 'O')) or (prevTag, Tag) in [('O', 'I'), ('E', 'E'), ('E', 'I'), ('O', 'E')]:
            chunkStart = True
        if Tag != 'O' and prevType != Type:
            chunkStart = True

        if (Tag in ('B','I') and nextTag in ('B','O')) or (Tag == 'E' and nextTag in ('E', 'I', 'O')):
            chunkEnd = True
        if Tag != 'O' and Type != nextType:
            chunkEnd = True

        if chunkStart:
            start_idx = idx
        if chunkEnd:
            end_idx = idx
            chunks.append((start_idx,end_idx,Type))
            start_idx,end_idx = 0,0
    return chunks