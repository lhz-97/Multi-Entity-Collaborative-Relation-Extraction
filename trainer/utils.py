import torch

from torch.autograd import Variable
import numpy as np
import sys
sys.path.append('..')
import config



from torch.nn import functional as F
class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)
    
def binScore(labels,predicts):
    TP=0
    FP=0
    FN=0
    TN=0
    for idx in range(len(labels)):
        if labels[idx]==1 and predicts[idx]==1:
            TP+=1
        if labels[idx]==1 and predicts[idx]==0:
            FN+=1
        if labels[idx]==0 and predicts[idx]==1:
            FP+=1
        if labels[idx]==0 and predicts[idx]==0:
            TN+=1
            
    recall=TP/(TP+FN+0.00001)
    precision=TP/(TP+FP+0.00001)
    f1=0.0
    if recall+precision>0:
        f1=2*recall*precision/(recall+precision)
    return recall,precision,f1

def showF1_all(idx,recall,precision,f1,re,label_num):
    if re:
        print('re label:{},f1:{} ,p:{}, r:{}, num:{}'.format(config.id2re[idx],f1,precision,recall,label_num))
    if not re:
        print('ner label:{},f1:{} ,p:{}, r:{}, num:{}'.format(config.id2ner[idx],f1,precision,recall,label_num))


def score(labels,predicts,re):
    if not re:
        ner_recall=[]
        ner_precision=[]
        for label in range(2,len(config.NER_LABEL_TO_ID)):
            nerlabel=[0]*len(labels)
            nerpredicts=[0]*len(labels)
            for idx in range(len(labels)):
                if labels[idx]==label:
                    nerlabel[idx]=1
                if predicts[idx]==label:
                    nerpredicts[idx]=1
            label_num=sum(nerlabel)
            recall,precision,f1=binScore(nerlabel,nerpredicts)
            showF1_all(label,recall,precision,f1,re,label_num)
            ner_recall.append(recall)
            ner_precision.append(precision)
        meanrecal=np.mean(ner_recall)
        meanprecison=np.mean(ner_precision)

        f1_score=2*meanprecison*meanrecal/(meanrecal+meanprecison)
    else:
        re_recall=[]
        re_precision=[]
        for label in range(1,len(config.RE_LABEL_TO_ID)):
            relabel=[0]*len(labels)
            repredicts=[0]*len(labels)
            for idx in range(len(labels)):
                if labels[idx]==label:
                    relabel[idx]=1
                if predicts[idx]==label:
                    repredicts[idx]=1
            
            label_num=sum(relabel)
            recall,precision,f1=binScore(relabel,repredicts)
            showF1_all(label,recall,precision,f1,re,label_num)
            re_recall.append(recall)
            re_precision.append(precision)
        meanrecal=np.mean(re_recall)
        meanprecison=np.mean(re_precision)


        f1_score=2*meanprecison*meanrecal/(meanrecal+meanprecison)
    return meanprecison,meanrecal,f1_score


def ner_tokenizer(batch):
    labels = Variable(batch[2].long())
    tokens = batch[0].long()
    lens = batch[1]
    return tokens, lens, labels

def get_pair(lens):
    pairs = []
    for i in range(lens):
        for j in range(lens):
            pairs.append((i, j))
    return pairs


def getadj(token_len,max_len):
    pairs = get_pair(token_len)
    max_len = max_len * max_len + max_len
    adj = torch.zeros([max_len, max_len])
    
    for pair in pairs:
        rel = (pair[0] + 1) * token_len + pair[1]
        adj[pair[0]][rel] = 1
        adj[rel][pair[1]] = 1
    
    return adj

all_adjs=[]

def searchadj(token_len,max_len):
    if all_adjs==[]:
        for l in range(91):
            all_adjs.append(getadj(l,l))
    max_len = max_len * max_len + max_len
    adj = torch.zeros([max_len, max_len])
    le=token_len*token_len+token_len
    adj= F.pad(all_adjs[token_len],(0,max_len-le,0,max_len-le))
    return adj




def getmasks(lens,max_len):
    max_len = max_len * max_len + max_len
    masks=torch.ones(size=(len(lens),max_len)).bool()
    for idx,l in enumerate(lens):
        masklen=l*l+l
        masks[idx][0:masklen]=False
    return masks


def adjust_learning_rate(optimizer, epoch,lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.9 ** (epoch // 6))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        