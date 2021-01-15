import torch
import numpy as np
import sys
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
sys.path.append('..')
import config
import torch.nn as nn
    
def get_long_tensor(tokens_list):
    batch_size=len(tokens_list)
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_batch_col(batch,idx):
    res=[]
    for b in batch:
        res.append(b[idx])
    return res

def get_deep_batch_col(batch,idx):
    res=[]
    for b in batch:
        for bb in b[idx]:
            res.append(bb)
    return res
def get_deep_deep_batch_col(batch,idx):
    res=[]
    for b in batch:
        for bb in b[idx]:
            for bbb in bb:
                res.append(bbb)
    return res


def get_re_inputs(batch,idx):
    res=[]
    for b in batch:
        if len(b[idx])!=0:
            if res==[]:
                res=[[] for i in range(len(b[idx][0]))]
            for i in range(len(b[idx])):
                for j in range(len(b[idx][i])):
                    if res[j]==[]:
                        res[j]=b[idx][i][j]
                    else:
                        res[j]=torch.cat((res[j],b[idx][i][j]),dim=0)
    return res




def simple_collate_fn(batch):
    tokens=get_long_tensor(get_batch_col(batch,0))
    sent_len=torch.tensor(get_batch_col(batch,1))
    ner_labels=get_long_tensor(get_batch_col(batch,2))
    re_nums=torch.tensor(get_batch_col(batch,3))
    re_inputs=get_re_inputs(batch,4)
    re_labels=torch.tensor(get_deep_batch_col(batch,5))
    
    return [tokens,sent_len,ner_labels,re_nums,re_inputs,re_labels]
    
    

def collate_fn(batch):
    tokens=get_long_tensor(get_batch_col(batch,0))
    sent_len=torch.tensor(get_batch_col(batch,1))
    ner_labels=get_long_tensor(get_batch_col(batch,2))
    re_nums=torch.tensor(get_batch_col(batch,3))
    re_inputs=get_re_inputs(batch,4)
    re_labels=torch.tensor(get_deep_batch_col(batch,5))


    gcn_inputs=get_re_inputs(batch,6)

    re_pairs=[]
    for i in range(len(batch)):
        if len(batch[i][7])==0:
            re_pairs.append([])
        else:
            pair=[]
            for tmp in batch[i][7]:
                pair.append(tmp)
            re_pairs.append(pair)
    return [tokens,sent_len,ner_labels,re_nums,re_inputs,re_labels,gcn_inputs,re_pairs]
 



    
