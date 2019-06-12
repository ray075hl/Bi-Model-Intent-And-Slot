import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import config as cfg


class IntentEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(IntentEncoder, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.lstm = nn.LSTM(embedding_size, hidden_size, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        # print('enc :', x.size())
        return x


class IntentDecoder(nn.Module):
    def __init__(self, hidden_size, num_of_intent):
        super(IntentDecoder, self).__init__()

        self.decoder = nn.LSTM(hidden_size*2, hidden_size, 1, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
                                     nn.Linear(in_features=hidden_size, out_features=hidden_size*2),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size*2, num_of_intent),
                                    )

        self.mix = nn.Linear(hidden_size*4, hidden_size*2)
        # print('hidden size: ', hidden_size)

    def forward(self, hi, hs, real_len):
        batch = hi.size()[0]
        h = torch.cat((hi, hs), dim=-1)
        # print(h.size())
        x = self.mix(h)
        x = F.relu(x)

        hidden_state, last_state = self.decoder(x)
        # print(last_state[0].size(), last_state[0].squeeze().size())
        # print(hidden_state.size(), real_len.size())
        index = torch.arange(batch).long().to(cfg.device)

        state_ = hidden_state[index, real_len, :]
        logit_intent = self.classifier(state_.squeeze())

        return logit_intent


class SlotEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(SlotEncoder, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.lstm = nn.LSTM(embedding_size, hidden_size, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)

        return x



class SlotDecoder(nn.Module):
    def __init__(self, hidden_size, num_of_slot):
        super(SlotDecoder, self).__init__()

        self.decoder = nn.LSTM(hidden_size * 2, hidden_size, 1, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
                                     nn.Linear(hidden_size, hidden_size*2),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size*2, num_of_slot),
                                    )

        self.mix = nn.Linear(hidden_size * 4, hidden_size * 2)

    def forward(self, hs, hi):

        h = torch.cat((hs, hi), dim=-1)
        x = self.mix(h)
        x = F.relu(x)

        slot_hidden, _ = self.decoder(x)
        logit_slot = self.classifier(slot_hidden)

        return logit_slot


class JointModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_of_intent, num_of_slot, maxlen, batch_size):
        super(JointModel, self).__init__()

        self.intent_enc = IntentEncoder(input_size, embedding_size, hidden_size)
        self.intent_dec = IntentDecoder(hidden_size, num_of_intent)

        self.slot_enc = SlotEncoder(input_size, embedding_size, hidden_size)
        self.slot_dec = SlotDecoder(hidden_size, num_of_slot)

        self.intent_share_hidden = torch.zeros(batch_size, maxlen, hidden_size*2).to(cfg.device)
        self.slot_share_hidden = torch.zeros(batch_size, maxlen, hidden_size*2).to(cfg.device)

    def forward(self, x):
        pass

    # def forward(self, x):
    #
    #
    #
    #     # xi = self.intent_enc(x)
    #     # if self.slot_share_hidden is None:
    #     #     logit_intent = self.intent_dec(xi, xi)
    #     # else:
    #     #     logit_intent = self.intent_dec(xi, self.slot_share_hidden)
    #     #
    #     # xs = self.slot_enc(x)
    #     # if self.intent_share_hidden is None:
    #     #     logit_slot = self.slot_dec(xs, xs)
    #     # else:
    #     #     logit_slot = self.slot_dec(xs, self.intent_share_hidden)
    #     #
    #     # self.slot_share_hidden = xs.clone()
    #     # self.intent_share_hidden = xi.clone()
    #     #
    #     # return logit_intent, logit_slot



