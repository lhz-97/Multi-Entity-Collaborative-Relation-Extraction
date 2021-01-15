import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .utils import *
import numpy as np
from torch.nn import functional as F

from numpy import *
import pickle
import sys
import json
sys.path.append('..')
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from data.dataloader import AceDataSet
from torch.nn.utils.rnn import pad_sequence
import config

id2word={}
for item in config.word2id:
    id2word[config.word2id[item]]=item
id2word[len(id2word)]='  XSSSS  '
id2word[len(id2word)]='  XEEEE  '
id2word[len(id2word)]='  OSSSS  '
id2word[len(id2word)]='  OEEEE  '
def print_case(tokens,label,predict,file1):
    sentss=[]
    
    
    for idx in range(len(tokens)):
        sentences=[]
        for t in tokens[idx]:
            sentences.append(id2word[int(t)])
        sentences=' '.join(sentences)
        sentss.append(sentences)
    return sentss
'''
        print('sentence:\t{}\t'.format(sentences),file=file1,end='')
        print('labels:\t{}\t'.format(config.id2re[int(label[idx])]),file=file1,end='')
        print('predicts:\t{}\t'.format(config.id2re[int(predict[idx])]),file=file1)
'''

def accN(score,pred,label):
    d=torch.argsort(-score)
    acc2=0
    acc4=0
    acc6=0
    accall=0
    for idx in range(len(d)):
        if idx<2:
            if pred[idx]==label[idx]:
                acc2+=1
                acc4+=1
                acc6+=1
                accall+=1
        elif idx<4:
            if pred[idx]==label[idx]:
                acc4+=1
                acc6+=1
                accall+=1
        elif idx<6:
            if pred[idx]==label[idx]:
                acc6+=1
                accall+=1
        else:
            if pred[idx]==label[idx]:
                accall+=1
    result=[]
    result.append(acc2)
    result.append(acc4)
    result.append(acc6)
    result.append(accall)
    return result

class Trainer(object):
    def __init__(self,
                 ner_model,
                 re_model,
                 gcn_model,
                 use_gcn,
                 train_loader,
                 val_loader,
                 path,
                 ner,
                 re,
                 nerweight,
                 reweight,
                 warmup_step,
                 f1_score,
                 batch_size=32,
                 max_epoch=500,
                 lr=0.01,
                 weight_decay=1e-5,
                 opt2='sgd'):
        self.max_epoch = max_epoch
        self.ner_model = ner_model
        self.re_model = re_model
        self.gcn_model = gcn_model
        self.use_gcn = use_gcn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.max_f1=0
        self.warmup_step=warmup_step
        self.ner=ner
        self.re=re
        self.path=path
        self.f1_score=f1_score
        if nerweight!=None:
            self.criterion_ner = nn.CrossEntropyLoss(weight=nerweight.cuda())
            self.criterion_re = nn.CrossEntropyLoss(weight=reweight.cuda())
        else:
            self.criterion_ner = nn.CrossEntropyLoss()
            self.criterion_re = nn.CrossEntropyLoss()
        
        self.ner_lr=0.05
        self.re_lr=0.005
        if self.re=='bert':
            self.re_lr/=10
        self.gcn_lr=0.01
        self.pro_num=25

        if opt2 == 'sgd':
            self.ner_optimizer = optim.SGD(self.ner_model.parameters(),
                                           self.ner_lr,
                                           weight_decay=weight_decay)
            self.re_optimizer = optim.SGD(self.re_model.parameters(),
                                           self.re_lr,
                                          weight_decay=weight_decay)
            self.gcn_optimizer = optim.SGD(self.gcn_model.parameters(),
                                           self.gcn_lr,
                                           weight_decay=weight_decay)
        if torch.cuda.is_available():
            self.re_model.cuda()
            self.ner_model.cuda()
            self.gcn_model.cuda()

    def train_model(self, metric='acc'):
        if self.use_gcn:
            global_step = 0
            for epoch in range(self.max_epoch):

                self.re_model.train()
                self.ner_model.train()
                #load re\ner\gcn
                self.gcn_model.train()
                ner_loss_val = 0
                re_loss_val = 0
                adjust_learning_rate(self.gcn_optimizer,epoch,self.gcn_lr)
                adjust_learning_rate(self.re_optimizer,epoch,self.re_lr)
                adjust_learning_rate(self.ner_optimizer,epoch,self.ner_lr)
                if epoch<self.warmup_step:
                    adjust_learning_rate(self.re_optimizer,113,self.re_lr)
                    adjust_learning_rate(self.ner_optimizer,113,self.ner_lr)

                for iter, data in enumerate(self.train_loader):
                    if torch.cuda.is_available():
                        for i in range(len(data)):
                            try:
                                data[i] = data[i].cuda()
                            except:
                                pass
                    tokens, lens, ner_labels = ner_tokenizer(data)
                    max_len=max(lens)
                    if max_len>90:
                        continue

                    ner_logits=None
                    #ner model
                    if self.ner =='lstm':
                        ner_logits=self.ner_model.get_gcn_emb(tokens)
                    #re model
                    re_logits=None
                    re_pairs=None
                    if self.re =='pcnn' or self.re =='cnn'or self.re =='bert'or self.re =='att_bilstm':
                        re_pairs = data[7]
                        for i in range(len(data[6])):
                            data[6][i]=data[6][i].cuda()
                        re_logits=self.re_model.get_gcn_emb(*data[6])


                    batchs = []
                    re_idx = 0
                    adjs = None
                    if self.re =='pcnn':
                        re_logits=self.gcn_model.pcnn(re_logits)
                    for idx, sent_len in enumerate(lens):
                        re_len = (sent_len * sent_len)
                        one_case = torch.cat(
                            (ner_logits[idx][0:sent_len],
                             re_logits[re_idx:re_idx + re_len]),
                            dim=0)
                        re_idx += re_len
                        batchs.append(one_case)
                        
                        if adjs is None:
                            adjs=searchadj(sent_len,max_len).unsqueeze(0)
                        else:
                            adjs=torch.cat((adjs,searchadj(sent_len,max_len).unsqueeze(0)),dim=0)
                        
                    adjs = adjs.cuda()
                    batchs = pad_sequence(batchs, batch_first=True).cuda()
                    masks = getmasks(lens,max_len)
                    
                    new_re_logits=None
                    have_rel=False
                    if self.re =='pcnn' or self.re =='cnn'or self.re =='bert'or self.re =='att_bilstm':
                        re_label = data[5]
                        for i in range(len(data[4])):
                            data[4][i]=data[4][i].cuda()
                        if len(data[4]) != 0:
                            have_rel=True
                            new_re_logits=self.re_model.get_gcn_emb(*data[4])
                    if self.re =='pcnn':
                        new_re_logits=self.gcn_model.pcnn(new_re_logits)
                    ner_logits,re_logits = self.gcn_model(batchs,ner_logits,new_re_logits,masks,adjs,lens,re_pairs)

                
                    batch_size = ner_logits.size(0)
                    
                    ner_logits = ner_logits.view(batch_size * max_len, -1)
                    ner_labels = ner_labels.view(-1)

                    for idx,l in enumerate(lens):
                        ner_logits[max_len*idx+l:max_len*(idx+1)]=0
                    
                    ner_loss = self.criterion_ner(ner_logits, ner_labels)
                    
                    if have_rel:
                        re_loss=self.criterion_re(re_logits,re_label)
                        
                        loss=re_loss+ner_loss
                        loss.backward()
                        self.re_optimizer.step()
                        self.re_optimizer.zero_grad()
                        re_loss_val+=re_loss.item()
                    else:
                        ner_loss.backward()
                    
                    self.ner_optimizer.step()
                    self.ner_optimizer.zero_grad()
                    
                    self.gcn_optimizer.step()
                    self.gcn_optimizer.zero_grad()
                    ner_loss_val+=ner_loss.item()
                    if iter%20==0:
                        print('ner loss:{},re loss:{}'.format(ner_loss_val,re_loss_val))
                        ner_loss_val=0
                        re_loss_val=0
                path=self.path+'-1'
                if epoch%10==0:
                    if not os.path.exists('./saved_models/combine/re{}'.format(path)):
                        os.makedirs('./saved_models/combine/re{}'.format(path))
                        os.makedirs('./saved_models/combine/ner{}'.format(path))
                        os.makedirs('./saved_models/combine/gcn{}'.format(path))
                    torch.save({'state_dict': self.re_model.state_dict()}, './saved_models/combine/re{}/{}'.format(path,epoch))
                    torch.save({'state_dict': self.ner_model.state_dict()}, './saved_models/combine/ner{}/{}'.format(path,epoch))
                    torch.save({'state_dict': self.gcn_model.state_dict()}, './saved_models/combine/gcn{}/{}'.format(path,epoch))
                print('epoch:{}'.format(epoch))
                self.eval_model(epoch)
                if self.pro_num==0:
                    break
        else:
            ner_loss_val = 0
            re_loss_val = 0
            
            for epoch in range(self.max_epoch):
                self.re_model.train()
                self.ner_model.train()
                adjust_learning_rate(self.re_optimizer,epoch,self.re_lr)
                adjust_learning_rate(self.ner_optimizer,epoch,self.ner_lr)
                if epoch<self.warmup_step:
                    adjust_learning_rate(self.re_optimizer,100,self.re_lr)

                for iter, data in enumerate(self.train_loader):
                    if torch.cuda.is_available():
                        for i in range(len(data)):
                            try:
                                data[i] = data[i].cuda()
                            except:
                                pass
                    tokens, lens, ner_labels = ner_tokenizer(data)
                    if self.ner =='lstm':
                        ner_logit = self.ner_model(tokens)

                    batch_size, max_len = ner_logit.size(0), ner_logit.size(1)
                    ner_logit = ner_logit.view(batch_size * max_len, -1)
                    ner_labels = ner_labels.view(-1)
                    
                    for idx,l in enumerate(lens):
                        ner_logit[max_len*idx+l:max_len*(idx+1)]=0
                    
                    ner_loss = self.criterion_ner(ner_logit, ner_labels)
                    ner_loss_val += ner_loss.item()

                    have_rel=False
                    if self.re =='pcnn' or self.re =='cnn'or self.re =='bert'or self.re =='att_bilstm':
                        re_label = data[5]
                        for i in range(len(data[4])):
                            data[4][i]=data[4][i].cuda()
                            
                        if len(data[4])!= 0:
                            logits = self.re_model(*data[4])
                            have_rel=True
                    
                    if have_rel:
                        reloss = self.criterion_re(logits, re_label)
                        re_loss_val += reloss.item()
                        loss=ner_loss+reloss
                        loss.backward()
                        self.re_optimizer.step()
                        self.re_optimizer.zero_grad()
                    else:
                        ner_loss.backward()
                    self.ner_optimizer.step()
                    self.ner_optimizer.zero_grad()

                    self.gcn_optimizer.step()
                    self.gcn_optimizer.zero_grad()

                    if iter%20==0:
                        print('ner loss:{},re loss:{}'.format(ner_loss_val,re_loss_val))
                        ner_loss_val=0
                        re_loss_val=0

                path=self.path
                if epoch%10==0:
                    if not os.path.exists('./saved_models/re{}'.format(path)):
                        os.makedirs('./saved_models/re{}'.format(path))
                        os.makedirs('./saved_models/ner{}'.format(path))
                    torch.save({'state_dict': self.re_model.state_dict()}, './saved_models/re{}/{}'.format(path,epoch))
                    torch.save({'state_dict': self.ner_model.state_dict()}, './saved_models/ner{}/{}'.format(path,epoch))
                print('epoch:{}'.format(epoch))
                self.eval_model(epoch)
                if self.pro_num==0:
                    break
    
    def eval_model(self,epoch):
        if self.use_gcn:
            ner_labels=[]
            ner_predicts=[]
            re_labels=[]
            re_predicts=[]

            acc2=[]
            acc4=[]
            acc6=[]
            accall=[]
            
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            self.re_model.eval()
            self.ner_model.eval()
            self.gcn_model.eval()
            
            pressss=[]
            file1=open('{}-gcnlabel.txt'.format(config.re_model),'w')
            for iter, data in enumerate(self.val_loader):
                
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass

                tokens, lens, _ = ner_tokenizer(data)
                max_len=max(lens)
                if max_len>90:
                    continue

                ner_logits=None
                #ner model
                if self.ner =='lstm':
                    ner_logits=self.ner_model.get_gcn_emb(tokens)
                #re model
                re_logits=None
                re_pairs=None
                if self.re =='pcnn' or self.re =='cnn'or self.re =='bert'or self.re =='att_bilstm':
                    re_pairs = data[7]
                    for i in range(len(data[6])):
                        data[6][i]=data[6][i].cuda()
                    re_logits=self.re_model.get_gcn_emb(*data[6])
                if self.re =='pcnn':
                    re_logits=self.gcn_model.pcnn(re_logits)

                batchs = []
                re_idx = 0
                adjs = None

                for idx, sent_len in enumerate(lens):
                    re_len = (sent_len * sent_len)
                    one_case = torch.cat(
                        (ner_logits[idx][0:sent_len],
                            re_logits[re_idx:re_idx + re_len]),
                        dim=0)
                    re_idx += re_len
                    batchs.append(one_case)
                    
                    if adjs is None:
                        adjs=searchadj(sent_len,max_len).unsqueeze(0)
                    else:
                        adjs=torch.cat((adjs,searchadj(sent_len,max_len).unsqueeze(0)),dim=0)
                    
                adjs = adjs.cuda()
                batchs = pad_sequence(batchs, batch_first=True).cuda()
                masks = getmasks(lens,max_len)
                
                new_re_logits=None
                have_rel=False
                if self.re =='pcnn' or self.re =='cnn'or self.re =='bert'or self.re =='att_bilstm':
                    re_label = data[5]
                    for i in range(len(data[4])):
                        data[4][i]=data[4][i].cuda()
                    if len(data[4]) != 0:
                        have_rel=True
                        new_re_logits=self.re_model.get_gcn_emb(*data[4])

                if self.re =='pcnn':
                    new_re_logits=self.gcn_model.pcnn(new_re_logits)

                ner_logits,re_logits = self.gcn_model(batchs,ner_logits,new_re_logits,masks,adjs,lens,re_pairs)

                batch_size = ner_logits.size(0)
                
                ner_logits = ner_logits.view(batch_size * max_len, -1)


                for idx,l in enumerate(lens):
                    ner_logits[max_len*idx+l:max_len*(idx+1)]=0
                
                #ner_acc
                ner_predict = np.argmax(ner_logits.data.cpu().numpy(),
                                    axis=1).tolist()
                batch_size = min(self.batch_size, len(data[1]))
                
                ner_label = Variable(data[2].long()).cpu()
                ner_predict = torch.tensor(ner_predict).view(batch_size, -1)

                if have_rel:
                    
                    scores, pred = re_logits.max(-1)


                    result=accN(scores,pred,re_label)

                    if scores.size(0)>=2:
                        acc2.append(result[0]/2)

                    if scores.size(0)>=4:
                        acc4.append(result[1]/4)

                    if scores.size(0)>=6:
                        acc6.append(result[2]/6)

                    accall.append(result[3]/scores.size(0))
                    '''
                    re_pred=pred
                    for re in re_label:
                        re_labels.append(re.item())
                    
                    for re in re_pred:
                        re_predicts.append(re.item())
                    '''
                    tmplabel=[]
                    tmppre=[]
                    for re in re_label:
                        re_labels.append(re.item())
                        tmplabel.append(re.item())
                    for re in pred:
                        re_predicts.append(re.item())
                        tmppre.append(re.item())
                    
                    pressss.append(tmppre)

                for ner_sent in ner_label:
                    for ner in ner_sent:
                        if ner == 0:
                            break
                        ner_labels.append(ner.item())
                
                for idx,ner_sent in enumerate(ner_predict):
                    l=lens[idx]
                    for ner in ner_sent:
                        if l==0:
                            break
                        ner_predicts.append(ner.item())
                        l-=1

                if iter%20==0:
                    print(iter)
                
                
            json.dump(pressss,file1)
            
            new_ner_labels=[]
            new_ner_predicts=[]

            for idx,ner in enumerate(ner_labels):
                if ner == 1 :
                    continue
                else:
                    new_ner_labels.append(ner)
                    new_ner_predicts.append(ner_predicts[idx])
            
            new_re_labels=[]
            new_re_predicts=[]
            for idx,re in enumerate(re_labels):
                if re == 0 and re_predicts[idx]==0:
                    continue
                else:
                    new_re_labels.append(re)
                    new_re_predicts.append(re_predicts[idx])


            print('acc@2:{},acc@4:{},acc@6:{},acc@all:{}'.format(mean(acc2),mean(acc4),mean(acc6),mean(accall)))
            if self.f1_score=='marco'or self.f1_score=='both':
                ner_p,ner_r,ner_f1=score(new_ner_labels, new_ner_predicts,False)
                ner_acc = accuracy_score(new_ner_labels,new_ner_predicts)
                print('marco all class ner f1: {}, p: {}, r: {}, acc: {}'.format(ner_f1,ner_p,ner_r,ner_acc))
                
                re_p,re_r,re_f1=score(new_re_labels, new_re_predicts,True)
                re_acc = accuracy_score(new_re_labels,new_re_predicts)
                print('marco all class re f1: {}, p: {}, r: {}, acc: {}'.format(re_f1,re_p,re_r,re_acc))
            if self.f1_score=='micro'or self.f1_score=='both':
                ner_positive=[i for i in range(2,len(config.id2ner))]
                ner_f1=f1_score(new_ner_labels,new_ner_predicts,labels=ner_positive,average='micro')
                ner_r=recall_score(new_ner_labels,new_ner_predicts,labels=ner_positive,average='micro')
                ner_p=precision_score(new_ner_labels,new_ner_predicts,labels=ner_positive,average='micro')
                
                ner_acc = accuracy_score(new_ner_labels,new_ner_predicts)
                print('mirco all class ner f1: {}, p: {}, r: {}, acc: {}'.format(ner_f1,ner_p,ner_r,ner_acc))
                re_positive=[i for i in range(1,len(config.id2re))]
                re_f1=f1_score(new_re_labels,new_re_predicts,labels=re_positive,average='micro')
                re_r=recall_score(new_re_labels,new_re_predicts,labels=re_positive,average='micro')
                re_p=precision_score(new_re_labels,new_re_predicts,labels=re_positive,average='micro')
                re_acc = accuracy_score(new_re_labels,new_re_predicts)
                print('mirco all class re f1: {}, p: {}, r: {}, acc: {}'.format(re_f1,re_p,re_r,re_acc))


            if ner_f1+re_f1>self.max_f1:
                self.max_f1=ner_f1+re_f1
                path=self.path+'-1'
                if not os.path.exists('./saved_models/combine/re{}'.format(path)):
                    os.makedirs('./saved_models/combine/re{}'.format(path))
                    os.makedirs('./saved_models/combine/ner{}'.format(path))
                    os.makedirs('./saved_models/combine/gcn{}'.format(path))

                torch.save({'state_dict': self.re_model.state_dict()}, './saved_models/combine/re{}/{}'.format(path,'best'))
                torch.save({'state_dict': self.ner_model.state_dict()}, './saved_models/combine/ner{}/{}'.format(path,'best'))
                torch.save({'state_dict': self.gcn_model.state_dict()}, './saved_models/combine/gcn{}/{}'.format(path,'best'))
                
                print('best all class ner f1: {}, p: {}, r: {}, acc: {}'.format(ner_f1,ner_p,ner_r,ner_acc))
                print('best all class re f1: {}, p: {}, r: {}, acc: {}'.format(re_f1,re_p,re_r,re_acc))

                self.pro_num=25
            else:
                self.pro_num-=1

        else:
            ner_labels=[]
            ner_predicts=[]
            re_labels=[]
            re_predicts=[]
            self.re_model.eval()
            self.ner_model.eval()
            
            acc2=[]
            acc4=[]
            acc6=[]
            accall=[]
            file1=open('{}-sigpred.txt'.format(config.re_model),'w')
            pressss=[]
            
            for iter, data in enumerate(self.val_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                tokens, lens, _ = ner_tokenizer(data)
                if self.ner =='lstm':
                    logits = self.ner_model(tokens)
                if iter%100==0:
                    print(iter)

                batch_size, max_len = logits.size(0), logits.size(1)
                logits = logits.view(batch_size * max_len, -1)
                
                for idx,l in enumerate(lens):
                    logits[max_len*idx+l:max_len*(idx+1)]=0


                predict = np.argmax(logits.data.cpu().numpy(),
                                    axis=1).tolist()
                batch_size = min(self.batch_size, len(data[1]))
                ner_label = Variable(data[2].long()).cpu()

                predict = torch.tensor(predict).view(batch_size, -1)

                have_rel=False
                if self.re =='pcnn' or self.re =='cnn'or self.re =='bert'or self.re =='att_bilstm':
                    re_label = data[5]
                    for i in range(len(data[4])):
                        data[4][i]=data[4][i].cuda()
                        
                    if len(data[4])!= 0:
                        logits = self.re_model(*data[4])
                        have_rel=True

                if have_rel:
                    scores, pred = logits.max(-1)

                    result=accN(scores,pred,re_label)

                    if scores.size(0)>=2:
                        acc2.append(result[0]/2)

                    if scores.size(0)>=4:
                        acc4.append(result[1]/4)

                    if scores.size(0)>=6:
                        acc6.append(result[2]/6)

                    accall.append(result[3]/scores.size(0))
                    tmplabel=[]
                    tmppre=[]
                    for re in re_label:
                        re_labels.append(re.item())
                        tmplabel.append(re.item())
                    for re in pred:
                        re_predicts.append(re.item())
                        tmppre.append(re.item())
                    
                    pressss.append(tmppre)

                for ner_sent in ner_label:
                    for ner in ner_sent:
                        if ner == 0:
                            break
                        ner_labels.append(ner.item())
                
                for idx,ner_sent in enumerate(predict):
                    l=lens[idx]
                    for ner in ner_sent:
                        if l==0:
                            break
                        ner_predicts.append(ner.item())
                        l-=1
            
            json.dump(pressss,file1)
            new_ner_labels=[]
            new_ner_predicts=[]
            
            for idx,ner in enumerate(ner_labels):
                if (ner == 1 and ner_predicts[idx]==1):
                    continue
                else:
                    new_ner_labels.append(ner)
                    new_ner_predicts.append(ner_predicts[idx])


            for idx,ner in enumerate(ner_labels):
                if ner == 1:
                    continue
                else:
                    new_ner_labels.append(ner)
                    new_ner_predicts.append(ner_predicts[idx])
            
            
            new_re_labels=[]
            new_re_predicts=[]

            for idx,re in enumerate(re_labels):
                if re == 0 and re_predicts[idx]==0:
                    continue
                else:
                    new_re_labels.append(re)
                    new_re_predicts.append(re_predicts[idx])


            print('acc@2:{},acc@4:{},acc@6:{},acc@all:{}'.format(mean(acc2),mean(acc4),mean(acc6),mean(accall)))
            if self.f1_score=='marco' or self.f1_score=='both':
                ner_p,ner_r,ner_f1=score(new_ner_labels, new_ner_predicts,False)
                ner_acc = accuracy_score(new_ner_labels,new_ner_predicts)
                print('marco all class ner f1: {}, p: {}, r: {}, acc: {}'.format(ner_f1,ner_p,ner_r,ner_acc))

                re_p,re_r,re_f1=score(new_re_labels, new_re_predicts,True)
                re_acc = accuracy_score(new_re_labels,new_re_predicts)
                print('marco all class re f1: {}, p: {}, r: {}, acc: {}'.format(re_f1,re_p,re_r,re_acc))
            if self.f1_score=='micro'or self.f1_score=='both':
                ner_positive=[i for i in range(2,len(config.id2ner))]
                ner_f1=f1_score(new_ner_labels,new_ner_predicts,labels=ner_positive,average='micro')
                ner_r=recall_score(new_ner_labels,new_ner_predicts,labels=ner_positive,average='micro')
                ner_p=precision_score(new_ner_labels,new_ner_predicts,labels=ner_positive,average='micro')
                
                ner_acc = accuracy_score(new_ner_labels,new_ner_predicts)
                print('mirco all class ner f1: {}, p: {}, r: {}, acc: {}'.format(ner_f1,ner_p,ner_r,ner_acc))
                re_positive=[i for i in range(1,len(config.id2re))]
                re_f1=f1_score(new_re_labels,new_re_predicts,labels=re_positive,average='micro')
                re_r=recall_score(new_re_labels,new_re_predicts,labels=re_positive,average='micro')
                re_p=precision_score(new_re_labels,new_re_predicts,labels=re_positive,average='micro')
                
                re_acc = accuracy_score(new_re_labels,new_re_predicts)
                print('mirco all class re f1: {}, p: {}, r: {}, acc: {}'.format(re_f1,re_p,re_r,re_acc))

            if ner_f1+re_f1>self.max_f1:
                self.max_f1=ner_f1+re_f1
                path=self.path
                
                torch.save({'state_dict': self.re_model.state_dict()}, './saved_models/re{}/{}'.format(path,'best'))
                torch.save({'state_dict': self.ner_model.state_dict()}, './saved_models/ner{}/{}'.format(path,'best'))
                
                print('best all class ner f1: {}, p: {}, r: {}, acc: {}'.format(ner_f1,ner_p,ner_r,ner_acc))
                print('best all class re f1: {}, p: {}, r: {}, acc: {}'.format(re_f1,re_p,re_r,re_acc))

                self.pro_num=25
            else:
                self.pro_num-=1
        
        
        
        
        