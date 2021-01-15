#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from .word_tokenizer import WordTokenizer

class Att_BLSTM(nn.Module):
    def __init__(self,word2id, word2vec,max_length=128,word_size=50, word_vec_len=10000,hidden_size=100, layers_num=1,dropout=0.3):
        super().__init__()
        self.word2id=word2id
        self.word_vec = word2vec
        # hyper parameters and others
        self.max_len = max_length
        self.word_dim = word_size
        self.hidden_size = hidden_size
        self.layers_num = layers_num
        self.emb_dropout_value = dropout
        self.lstm_dropout_value = dropout
        self.linear_dropout_value = dropout
        self.word_vec_len=word_vec_len
        self.tokenizer = WordTokenizer(vocab=self.word2id, unk_token="[UNK]")
        embeddings=torch.cat((torch.from_numpy(self.word_vec),torch.from_numpy(np.random.normal(0, 1, (4, self.word_dim)))),dim=0)
        
        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=embeddings,
            freeze=False,
        )
        self.lstm = nn.LSTM(
            input_size=self.word_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layers_num,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        self.tanh = nn.Tanh()
        self.emb_dropout = nn.Dropout(self.emb_dropout_value)
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)

        self.att_weight = nn.Parameter(torch.randn(1, self.hidden_size, 1))
        

    def lstm_layer(self, x, mask):
        lengths = torch.sum(mask.gt(0), dim=-1)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        h, (_, _) = self.lstm(x.float())
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)
        h = h.view(-1, self.max_len, 2, self.hidden_size)
        h = torch.sum(h, dim=2)  # B*L*H
        return h

    def attention_layer(self, h, mask):
        att_weight = self.att_weight.expand(mask.shape[0], -1, -1)  # B*H*1
        att_score = torch.bmm(self.tanh(h), att_weight)  # B*L*H  *  B*H*1 -> B*L*1

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=-1)  # B*L*1
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))  # B*L*1
        att_weight = F.softmax(att_score, dim=1)  # B*L*1

        reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1)  # B*H*L *  B*L*1 -> B*H*1 -> B*H
        reps = self.tanh(reps)  # B*H
        return reps

    def forward(self, token,mask):
        token = token
        mask = mask
        emb = self.word_embedding(token)  # B*L*word_dim
        emb = self.emb_dropout(emb)
        h = self.lstm_layer(emb, mask)  # B*L*H

        h = self.lstm_dropout(h)
        reps = self.attention_layer(h, mask)  # B*reps

        return reps


    def tokenize(self, item):
        tokens=item['token']
        token = self.tokenizer.convert_tokens_to_ids(tokens, unk_id = self.word2id['[UNK]'])
        pos_head = item['Arg-1_position']
        pos_tail = item['Arg-2_position']
        e1s=self.word_vec_len
        e1e=self.word_vec_len+1
        e2s=self.word_vec_len+2
        e2e=self.word_vec_len+3
        new_token=[]
        for i in range(len(token)):
            if i==pos_head[0]:
                new_token.append(e1s)
            if i==pos_tail[0]:
                new_token.append(e2s)
            new_token.append(token[i])
            if i==pos_tail[1]:
                new_token.append(e2e)
            if i==pos_head[1]:
                new_token.append(e1e)


        mask = [1] * len(new_token)

        length = min(self.max_len, len(new_token))
        mask = mask[:length]

        new_token=new_token[:length]
        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                new_token.append(0)
                
        return torch.tensor(new_token).long().unsqueeze(0),torch.tensor(mask).long().unsqueeze(0)
    
    def gcn_tokenize(self, tokens,pos_head,pos_tail):

        tokens=tokens
        token = self.tokenizer.convert_tokens_to_ids(tokens, unk_id = self.word2id['[UNK]'])
        e1s=self.word_vec_len
        e1e=self.word_vec_len+1
        e2s=self.word_vec_len+2
        e2e=self.word_vec_len+3
        new_token=[]
        for i in range(len(token)):
            if i==pos_head[0]:
                new_token.append(e1s)
            if i==pos_tail[0]:
                new_token.append(e2s)
            new_token.append(token[i])
            if i==pos_tail[1]:
                new_token.append(e2e)
            if i==pos_head[1]:
                new_token.append(e1e)


        mask = [1] * len(new_token)

        length = min(self.max_len, len(new_token))
        mask = mask[:length]

        new_token=new_token[:length]
        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                new_token.append(0)
                
        return torch.tensor(new_token).long().unsqueeze(0),torch.tensor(mask).long().unsqueeze(0)
    