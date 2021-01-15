import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import random

import sys
sys.path.append('..')
import config

from utils.function import *

class BiLSTM(nn.Module):
    def __init__(self,embed_num,embed_dim,lstm_hiddens,lstm_layers,label_num,dropout):
        super(BiLSTM,self).__init__()
        self.embed_num=embed_num
        self.embed_dim=embed_dim
        self.label_num=label_num
        self.lstm_layers=lstm_layers
        self.dropout=dropout
        self.lstm_hiddens=lstm_hiddens
        self.embed=nn.Embedding(embed_num,embed_dim,padding_idx=config.padding_idx)
        
        embedding=config.word2vec
        if config.pretrained_embed:
            self.embed.weight.data.copy_(torch.from_numpy(embedding))
        else:
            self.embed.weight=init_embedding(self.embed.weight)
        self.dropout=nn.Dropout(self.dropout)
        
        self.bilstm = nn.LSTM(input_size=embed_dim,hidden_size=lstm_hiddens,num_layers=lstm_layers,bidirectional=True,batch_first=True,bias=True)
        self.linear = nn.Linear(in_features=self.lstm_hiddens*2,out_features=label_num)

    def forward(self,word):
        
        x=self.embed(word)
        x=self.dropout(x)
        x,_=self.bilstm(x)

        x=torch.tanh(x)
        logit=self.linear(x)
        
        return logit

    def forward1(self,word,lens):
        
        word = pack_padded_sequence(word, lens, batch_first=True, enforce_sorted=False)
        x=self.embed(word)
        x=self.dropout(x)
        x,_=self.bilstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0.0)

        x=torch.tanh(x)
        logit=self.linear(x)
        
        return logit
    def get_gcn_emb(self,word):
        x=self.embed(word)
        x=self.dropout(x)
        x,_=self.bilstm(x)
        x=torch.tanh(x)
        return x









