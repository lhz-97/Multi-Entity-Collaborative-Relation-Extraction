import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import utils
import random
import math
from torch.nn.parameter import Parameter
import sys
sys.path.append('..')
import config
#from utils.function import *

from torch.autograd import Variable
class GCNClassifier(nn.Module):
    def __init__(self,lstm_hiddens,ner_label_num,encode_hidden_size,re_label_num):
        super(GCNClassifier,self).__init__()
        
        self.gcn_model = GCN(config.gcn_embedding_dim,False,config.gcn_hiddens,1,0.5,config.gcn_layers)
        
        self.gcn_model.cuda()
        self.dropout=nn.Dropout()
        
        self.dense_ner_linear = nn.Linear(in_features=lstm_hiddens*2*2,out_features=ner_label_num)
        
        self.dense_re_linear = nn.Linear(encode_hidden_size*2, re_label_num)
        

    def forward(self,inputs,ner,re,masks,adj,sent_len,re_pairs):
        inputs=self.dropout(inputs)
        logits,_=self.gcn_model(inputs,masks,adj)
        ner_logits = []
        #max_len=max(sent_len)
        #ner_logits=logits[:][:max_len]
        for idx,l in enumerate(sent_len):
            ner_logits.append(logits[idx][:l])
        ner_logits=pad_sequence(ner_logits,batch_first=True)

        ner_logits=torch.cat((ner,ner_logits),dim=2)
        
        ner_logits=self.dropout(ner_logits)
        ner_logits = self.dense_ner_linear(ner_logits)
        
        re_logits=None
        sent_id=0
        for pair in re_pairs:
            l=sent_len[sent_id]

            if len(pair)!=0:
                for head,tail in pair:
                    rel_num=0
                    avg_logits=0
                    for i in range(head[0],head[1]+1):
                        for j in range(tail[0],tail[1]+1):
                            rel_num+=1
                            avg_logits+=logits[sent_id][self.get_re(l,i,j)]
                    avg_logits=(avg_logits/rel_num).unsqueeze(0)
                    if re_logits is None:
                        re_logits=avg_logits
                    else:
                        re_logits=torch.cat((re_logits,avg_logits),dim=0)

            sent_id+=1
        if re_logits is not None:
            re_logits=re_logits.cuda()
            re_logits=torch.cat((re,re_logits),dim=1)
            re_logits=self.dropout(re_logits)
            re_logits = self.dense_re_linear(re_logits)
            
        return ner_logits,re_logits
    
    def get_re(self,sent_len,head,tail):
        index=(head+1)*sent_len+tail
        return index
    
    
    
    
class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, in_dim,use_rnn, rnn_hidden,rnn_layers,dropout,num_layers):
        super(GCN, self).__init__()
        self.layers = num_layers
        
        self.in_dim =in_dim
        # rnn layer
        self.use_rnn=use_rnn
        if use_rnn:
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size,rnn_hidden, rnn_layers, batch_first=True, \
                    dropout=dropout, bidirectional=True)
            self.in_dim = rnn_hidden * 2
            self.rnn_drop = nn.Dropout(dropout) # use on last layer output

        self.in_drop = nn.Dropout(dropout)
        self.gcn_drop = nn.Dropout(dropout)
        self.mem_dim=rnn_hidden*2
        self.rnn_hidden=rnn_hidden
        self.rnn_layers=rnn_layers
        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))
        
        self.output=nn.Linear(self.mem_dim*num_layers,self.mem_dim)

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        
        seq_lens = list(masks.data.eq(config.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.rnn_hidden, self.rnn_layers)
        
        
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True,enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs,masks,adj):
        embs=inputs
        #embs = self.in_drop(embs)
        # rnn layer
        if self.use_rnn:
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, inputs.size()[0]))
        else:
            gcn_inputs = embs
        gcn_outputs = None

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)


        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            if gcn_outputs is None:
                gcn_outputs=gAxW
            else:
                gcn_outputs=torch.cat((gcn_outputs,gAxW),dim=2)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        gcn_outputs=self.output(gcn_outputs)
        return gcn_outputs, mask


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

    








