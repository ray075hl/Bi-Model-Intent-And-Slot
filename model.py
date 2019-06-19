from data import word2index, index2word, slot2index, index2slot, intent2index, index2intent
from data import index_train, index_test

import torch 
import torch.nn as nn
import torch.nn.functional as F

from config import device

DROP_OUT = 0.2

# class SlotFilling(nn.Module):

#     def __init__(self, vocab_size=len(word2index), label_size=len(slot2index)):
#         super(SlotFilling, self).__init__()
#         embedding_dim = 300
#         hidden_size = 200
#         self.embeddding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size= hidden_size,\
#                             bidirectional= True, batch_first=True)
#         self.fc = nn.Linear(hidden_size*2, label_size)

#     def forward(self, x):
#         x = self.embeddding(x)
#         x, _ = self.lstm(x)
#         x = F.dropout(x, DROP_OUT)
#         outputs = self.fc(x)
#         return outputs


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
        self.mix = nn.Linear(hidden_size*4, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_size, label_size)

    def forward(self, x, hi):
        x = torch.cat((x, hi), dim=-1)
        x = self.mix(x)

        x = F.tanh(x)

        x, _ = self.lstm(x)
        x = F.dropout(x, DROP_OUT)
        res = self.fc(x)
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
        self.mix = nn.Linear(hidden_size*4, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=2)
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, label_size)
        self.fc = nn.Linear(hidden_size, label_size)

    def forward(self, x, hs, real_len):
        batch = x.size()[0]
        real_len = torch.tensor(real_len).to(device)
        x = torch.cat((x, hs), dim=-1)
        x = self.mix(x)

        x = F.tanh(x)

        x, last_state = self.lstm(x)
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

