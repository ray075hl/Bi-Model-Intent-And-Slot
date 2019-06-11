import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
from data import *
from model import *
import config as cfg
import utils

DEBUG = False

USE_CUDA = torch.cuda.is_available()


train_data, word2index, tag2index, intent2index = preprocessing('./data/atis-2.train.w-intent.iob', cfg.max_length)

if DEBUG:
    print(len(train_data))
    print(len(word2index))
    print(len(tag2index))
    print(intent2index)


my_model = JointModel(len(word2index), cfg.embedding_size,
                      cfg.hidden_size, len(intent2index), len(tag2index),
                      cfg.max_length, cfg.batch_size).to(cfg.device)

IntentLoss = nn.CrossEntropyLoss()
SlotLoss = nn.CrossEntropyLoss()

optim_intent = optim.Adam(my_model.parameters(), lr=cfg.learning_rate)
optim_slot = optim.Adam(my_model.parameters(), lr=cfg.learning_rate)


for epoch in range(cfg.total_epoch):
    losses = []
    for i, batch in enumerate(getBatch(cfg.batch_size, train_data)):

        # ----------------------------- papare data -------------------------
        x, y_1, y_2 = zip(*batch)   # sentence, slot label, intent label

        x = torch.cat(x)
        tag_target = torch.cat(y_1)

        slot_mask = utils.make_mask(tag_target, len(tag2index), cfg.max_length, cfg.device)

        intent_target = torch.cat(y_2)
        tag_target = utils.one_hot(tag_target, len(tag2index), cfg.max_length, cfg.device)
        intent_target = utils.one_hot(intent_target, len(intent2index), cfg.max_length, cfg.device)

        # ----------------------------- compute graph ------------------------

        hi = my_model.intent_enc(x)
        my_model.intent_share_hidden = hi.clone()
        intent_logits = my_model.intent_dec(hi, my_model.slot_share_hidden.detach())

        intent_loss = -1.0 * torch.sum(F.log_softmax(intent_logits, dim=-1) * intent_target) / len(intent2index)
        my_model.zero_grad()
        intent_loss.backward()
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), 5.0)
        optim_intent.step()

        # Asynchronous training
        hs = my_model.slot_enc(x)
        my_model.slot_share_hidden = hs.clone()
        slot_logits = my_model.slot_dec(hs, my_model.intent_share_hidden.detach())
        # print(slot_logits.size())

        slot_loss = -1.0 * torch.sum(utils.masked_log_softmax(slot_logits, slot_mask, dim=-1) * tag_target) / len(tag2index)
        my_model.zero_grad()
        slot_loss.backward()
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), 5.0)
        optim_slot.step()

    print(intent_loss.item(), slot_loss.item())

