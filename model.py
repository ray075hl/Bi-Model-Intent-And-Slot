from data import word2index, index2word, slot2index, index2slot, intent2index, index2intent
from data import index_train, index_test

import torch 
import torch.nn as nn
import torch.nn.functional as F

from config import device

DROP_OUT = 0.2


# Bi-model 

class slot_enc(nn.Module):
    def __init__(self, vocab_size=len(word2index)):
        super(slot_enc, self).__init__()
        embedding = 300
        hidden_size = 200
        self.embedding = nn.Embedding(vocab_size, embedding).to(device)
        # self.embedding.weight.data.uniform_(-1.0, 1.0)
        self.lstm = nn.LSTM(input_size=embedding, hidden_size= hidden_size, num_layers=2,\
                            bidirectional= True, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        x = F.dropout(x, DROP_OUT)
        x, _ = self.lstm(x)

        return x 


class slot_dec(nn.Module):
    def __init__(self, hidden_size= 200, label_size=len(slot2index)):
        super(slot_dec, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_size*5, hidden_size=hidden_size, num_layers=2)
        self.fc = nn.Linear(hidden_size, label_size)
        self.hidden_size = hidden_size
    def forward(self, x, hi):
        batch = x.size(0)
        length = x.size(1)
        dec_init_out = torch.zeros(batch, 1, self.hidden_size).to(device)
        hidden_state = (torch.zeros(2, 1, self.hidden_size).to(device), \
                        torch.zeros(2, 1, self.hidden_size).to(device))
        x = torch.cat((x, hi), dim=-1)
        x = x.transpose(1, 0) # 50 x batch x feature_size
        
        all_out = []
        for i in range(length):
            if i == 0:
                out, hidden_state = self.lstm(torch.cat((x[i].unsqueeze(1), dec_init_out), dim=-1), hidden_state)
            else:
                # print('out_size: ', out.size())
                # print('x size: ', x[i].size())
                out, hidden_state = self.lstm(torch.cat((x[i].unsqueeze(1), out), dim=-1), hidden_state)
            # print(out.size())
            all_out.append(out)
        output = torch.cat(all_out, dim=1) # 50 x batch x feature_size

        print(output.size())
        res = self.fc(output)
        return res 









class intent_enc(nn.Module):
    def __init__(self, vocab_size=len(word2index)):
        super(intent_enc, self).__init__()
        embedding = 300
        hidden_size = 200
        self.embedding = nn.Embedding(vocab_size, embedding).to(device)
        # self.embedding.weight.data.uniform_(-1.0, 1.0)
        self.lstm = nn.LSTM(input_size=embedding, hidden_size= hidden_size, num_layers=2,\
                            bidirectional= True, batch_first=True)
    
    def forward(self, x):
        x = self.embedding(x)
        x = F.dropout(x, DROP_OUT)
        x, _ = self.lstm(x)

        return x


class intent_dec(nn.Module):
    def __init__(self, hidden_size= 200, label_size=len(intent2index)):
        super(intent_dec, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_size*4, hidden_size=hidden_size, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_size, label_size)

    def forward(self, x, hs, real_len):
        batch = x.size()[0]
        real_len = torch.tensor(real_len).to(device)
        x = torch.cat((x, hs), dim=-1)

        x, _ = self.lstm(x)
        x= F.dropout(x, DROP_OUT)
        index = torch.arange(batch).long().to(device)
        state = x[index, real_len-1, :]
        
        res = self.fc(state.squeeze())
        return res
        
        # res = self.fc1(last_state[0])
        # res = F.relu(res)
        # res = self.fc2(res)
        # return res 


class Intent(nn.Module):
    def __init__(self):
        super(Intent, self).__init__()
        self.enc = intent_enc().to(device)
        self.dec = intent_dec().to(device)
        self.share_memory = torch.zeros(16, 50, 400).to(device)
    

class Slot(nn.Module):
    def __init__(self):
        super(Slot, self).__init__()
        self.enc = slot_enc().to(device)
        self.dec = slot_dec().to(device)
        self.share_memory = torch.zeros(16, 50, 400).to(device)

