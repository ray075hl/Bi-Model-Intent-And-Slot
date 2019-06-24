from model import *
 

from torch import optim
import numpy as np
import torch

import utils

from config import device
#from data_processing import train_data_list, test_data_list
from data2index import train_data, test_data
epoch_num = 50

slot_model = Slot().to(device)
intent_model = Intent().to(device)

slot_optimizer = optim.Adam(slot_model.parameters(), lr=0.001)
intent_optimizer = optim.Adam(intent_model.parameters(), lr=0.001)

best_correct_num = 0
best_epoch = -1
for epoch in range(epoch_num):
    slot_loss_history = []
    intent_loss_history = []
    for batch_index, data in enumerate(utils.get_batch(train_data)):
        # print(batch_index)
        sentence, real_len, slot_label, intent_label = data
        # print(sentence)
        mask = utils.make_mask(real_len).to(device)
        slot_optimizer.zero_grad()
        # print(sentence[0].shape, real_len.shape, slot_label.shape)
        x = torch.tensor(sentence).to(device)
        y_slot = torch.tensor(slot_label).to(device)
        y_slot = utils.one_hot(y_slot).to(device)
        y_intent = torch.tensor(intent_label).to(device)
        y_intent = utils.one_hot(y_intent, Num=18).to(device)
        # print(x.size(), y.size())
        hs = slot_model.enc(x)
        slot_model.share_memory = hs.clone()
        slot_logits = slot_model.dec(hs, intent_model.share_memory.detach())
        log_slot_logits = utils.masked_log_softmax(slot_logits, mask, dim=-1)
        slot_loss = -1.0*torch.sum(y_slot*log_slot_logits)
        slot_loss_history.append(slot_loss.item())
        slot_loss.backward()
        torch.nn.utils.clip_grad_norm_(slot_model.parameters(), 5.0)
        slot_optimizer.step()

        ##############################################
        intent_optimizer.zero_grad()
        hi = intent_model.enc(x)
        intent_model.share_memory = hi.clone()
        intent_logits = intent_model.dec(hi, slot_model.share_memory.detach(), real_len)
        log_intent_logits = F.log_softmax(intent_logits, dim=-1)
        intent_loss = -1.0*torch.sum(y_intent*log_intent_logits)
        intent_loss_history.append(intent_loss.item())
        intent_loss.backward()
        torch.nn.utils.clip_grad_norm_(intent_model.parameters(), 5.0)
        intent_optimizer.step()

        if batch_index % 100 == 0 and batch_index > 0:
            print('Slot loss: {:.4f} \t Intent loss: {:.4f}'.format(sum(slot_loss_history[-100:])/100.0, \
                sum(intent_loss_history[-100:])/100.0))

    # evaluation
    total_test = len(test_data)
    # print(total_test)
    correct_num = 0

    for batch_index, data_test in enumerate(utils.get_batch(test_data, batch_size=1)):
        sentence_test, real_len_test, slot_label_test, intent_label_test = data_test
        # print(sentence[0].shape, real_len.shape, slot_label.shape)
        x_test = torch.tensor(sentence_test).to(device)

        # slot model to generate hs
        hs_test = slot_model.enc(x_test)

        mask_test = utils.make_mask(real_len_test, batch=1).to(device)
        hi_test = intent_model.enc(x_test)
        # intent_logits_test = intent_model.dec(hi_test, torch.zeros(1, 50, 400).to(device), real_len_test)
        intent_logits_test = intent_model.dec(hi_test, hs_test, real_len_test)
        log_intent_logits_test = F.log_softmax(intent_logits_test, dim=-1)
        res_test = torch.argmax(log_intent_logits_test, dim=-1)
        

        if res_test.item() == intent_label_test[0]:
            correct_num += 1
        if correct_num > best_correct_num:
            best_correct_num = correct_num
            best_epoch = epoch
    
    print('*'*20)
    print('Epoch: [{}/{}], Intent Val Acc: {:.4f}'.format(epoch+1, epoch_num, 100.0*correct_num/total_test))
    print('*'*20)
    
    print('Best Intent Acc: {:.4f} at Epoch: [{}]'.format(best_correct_num/total_test, best_epoch+1))

