import json
import random
import torch
import numpy as np
import pickle
import sys
sys.path.append('..')
import config
import utils.constant as constant

dataset=''

def get_pair(lens):
    pairs=[]
    for i in range(lens):
        for j in range(lens):
            pairs.append(([i,i],[j,j]))

    return pairs

class AceDataSet(torch.utils.data.Dataset):
    def __init__(self,filename,encoder):
        super(AceDataSet,self).__init__()
        self.word2id=config.word2id
        json_file=open(filename,'r')
        self.data=json.load(json_file)
        self.encoder=encoder
        self.relabel2id=config.RE_LABEL_TO_ID
        self.ner_label2id=config.NER_LABEL_TO_ID
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self,index):
        tokens = torch.LongTensor(len(self.data[index]['token']))
        for i,word in enumerate(self.data[index]['token']):
            word=word.lower()
            if word in self.word2id:
                tokens[i]=self.word2id[word]
            else:
                tokens[i]=1
                
        ner_labels = torch.LongTensor(len(self.data[index]['entity']))
        
        for i,label in enumerate(self.data[index]['entity']):
            ner_labels[i]=self.ner_label2id[label]
        
        rel_labels = []

        re_inputs=[]

        re_num=0
        re_pairs=[]
        for relation_item in self.data[index]['relation']:
            rel_labels.append(self.relabel2id[relation_item['relation_type']])
            relation_item['token']=self.data[index]['token']
            head=relation_item['Arg-1_position']
            tail=relation_item['Arg-2_position']
            re_pairs.append((head,tail))
            re_inputs.append(self.encoder.tokenize(relation_item))
            re_num+=1
        

        gcn_inputs=[]
        for (head,tail) in get_pair(len(tokens)):
            gcn_inputs.append(self.encoder.gcn_tokenize(self.data[index]['token'],head,tail))
            
        return [tokens,len(tokens),ner_labels,re_num,re_inputs,rel_labels,gcn_inputs,re_pairs]



class SimpleAceDataSet(torch.utils.data.Dataset):
    def __init__(self,filename,encoder):
        super(SimpleAceDataSet,self).__init__()
        self.word2id=config.word2id
        json_file=open(filename,'r')
        self.data=json.load(json_file)
        self.encoder=encoder
        self.relabel2id=config.RE_LABEL_TO_ID
        self.ner_label2id=config.NER_LABEL_TO_ID
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self,index):
        tokens = torch.LongTensor(len(self.data[index]['token']))

        for i,word in enumerate(self.data[index]['token']):
            word=word.lower()
            if word in self.word2id:
                tokens[i]=self.word2id[word]
            else:
                tokens[i]=1
                
        ner_labels = torch.LongTensor(len(self.data[index]['entity']))
        self.ner_label2id=config.NER_LABEL_TO_ID
        for i,label in enumerate(self.data[index]['entity']):
            ner_labels[i]=self.ner_label2id[label]

        rel_labels = []
        re_inputs=[]

        re_num=0
        re_pairs=[]
        for relation_item in self.data[index]['relation']:
            rel_labels.append(self.relabel2id[relation_item['relation_type']])
            relation_item['token']=self.data[index]['token']
            head=relation_item['Arg-1_position']
            tail=relation_item['Arg-2_position']
            re_pairs.append((head,tail))
            re_inputs.append(self.encoder.tokenize(relation_item))
            re_num+=1

            
        return [tokens,len(tokens),ner_labels,re_num,re_inputs,rel_labels]

