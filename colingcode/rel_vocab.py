'''
prepare vocablary and initial word vectors.
'''
import json
import pickle
import argparse
import numpy as np
from collections import Counter

from utils import vocab, constant,helper
def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir',default='dataset/wiki',help='Ace directory.')
    parser.add_argument('--vocab_dir',default='dataset/wiki',help='Output directory')
    parser.add_argument('--glove_dir',default='/data/liuhz/paper/GCN_copy/dataset/glove',help='')
    parser.add_argument('--wv_file',default='glove.6B.100d.txt',help='Glove vector file')
    parser.add_argument('--wv_dim',type=int,default=100,help='Glove vector dimension')
    parser.add_argument('--min_freq',type=int,default=0,help='if >0 use min_freq as the cutoff.')
    parser.add_argument('lower',action='store_true',help='If specified, lowercase all words.')
    
    args=parser.parse_args()
    return args

def load_tokens(filename,data_dir):
    with open(filename) as infile:
        data=json.load(infile)
        tokens=[]
        for d in data:
            ts = d['token']
            tokens+=list(filter(lambda t:t!='<PAD>',ts))
    
    print('{} tokens from {} examples load from{}.'.format(len(tokens),len(data),filename))
    return tokens

def build_vocab(tokens,glove_vocab,min_freq,data_dir):
    counter=Counter(t for t in tokens)
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >=min_freq],key = counter.get, reverse =True)
    else:
        v = sorted([t for t in counter if t in glove_vocab],key = counter.get, reverse =True)
    v=constant.VOCAB_PREFIX + v
    print('vocab built with {}/{} words.'.format(len(v),len(counter)))
    return v

def count_oov(tokens,vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total ,total-matched





def main():
    #input file
    args=parse_args()
    train_file=args.data_dir+'/train.json'
    test_file=args.data_dir+'/test.json'
    wv_file=args.glove_dir+'/'+args.wv_file
    wv_dim=args.wv_dim
    
    #output file
    helper.ensure_dir(args.vocab_dir)
    vocab_file =args.vocab_dir+'/vocab.pkl'
    emb_file=args.vocab_dir+'/embedding.npy'
    
    #load files
    print('loading files...')
    train_tokens = load_tokens(train_file,args.data_dir)
    test_tokens = load_tokens(test_file,args.data_dir)
    if args.lower:
        train_tokens,test_tokens = [[t.lower() for t in tokens] for tokens in (train_tokens,test_tokens)]
    
    print('loading glove...')
    glove_vocab = vocab.load_glove_vocab(wv_file,wv_dim)
    
    print('{} words loaded from glove.'.format(len(glove_vocab)))
    
    print('building vocab...')
    v= build_vocab(train_tokens,glove_vocab,args.min_freq,args.data_dir)
    
    print('calculating oov...')
    datasets={'train':train_tokens,'test':test_tokens}
    
    for dname, d in datasets.items():
        total,oov = count_oov(d,v)
        print('{} ovv:{}/{} ({:.2f}%)'.format(dname,oov,total,oov*100.0/total))
        
    print('building embeddings...')
    embedding = vocab.build_embedding(wv_file,v,wv_dim)
    print('embedding size: {} x {}'.format(*embedding.shape))
    
    print('dumpomg to files ...')
    word2id={}
    for idx,word in enumerate(v):
        word2id[word]=idx
    with open(vocab_file,'wb') as outfile:
        pickle.dump((v,word2id),outfile)
    np.save(emb_file,embedding)
    print('all done.')
    with open(args.vocab_dir+'/word2id.json','w') as outfile:
        json.dump(word2id,outfile)
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    