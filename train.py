import sys, json
import torch
import os
import numpy as np
from encoders.pcnn_encoder import PCNNEncoder
from encoders.bert_encoder import BERTEncoder
from encoders.cnn_encoder import CNNEncoder
from encoders.att_blstm_encoder import Att_BLSTM
from model.softmax_nn import SoftmaxNN
from trainer.trainer import Trainer
import config
from data.dataloader import AceDataSet,SimpleAceDataSet
from torch.utils import data
import utils.function as function
import utils.constant as constant
from utils.vocab import Vocab

from model.gcn_model import GCNClassifier
import random
from model.ner_Bilstm_model import BiLSTM
import pickle

import argparse


randseed=1234
torch.manual_seed(1444)
np.random.seed(1444)
random.seed(1444)
torch.cuda.manual_seed_all(1444)#为所有GPU设置随机种子
warmup_step=0
if config.re_model=='pcnn':
    sentence_encoder = PCNNEncoder(token2id=config.word2id,
                            max_length=90,
                            word_size=100,
                            position_size=5,
                            hidden_size=100,
                            blank_padding=True,
                            kernel_size=3,
                            padding_size=1,
                            word2vec=config.word2vec,
                            dropout=0.5)
elif config.re_model=='bert':
    sentence_encoder = BERTEncoder(90,'.')

elif config.re_model=='cnn':
    sentence_encoder = CNNEncoder(token2id=config.word2id,
                        max_length=90,
                        word_size=100,
                        position_size=5,
                        hidden_size=100,
                        blank_padding=True,
                        kernel_size=3,
                        padding_size=1,
                        word2vec=config.word2vec,
                        dropout=0.5)
elif config.re_model=='att_bilstm':
    sentence_encoder = Att_BLSTM(word2id=config.word2id,
                        word2vec=config.word2vec,
                        max_length=90,
                        word_size=100,
                        word_vec_len=config.embed_num,
                        hidden_size=100,
                        layers_num=1,
                        dropout=0.3)


if config.ner_model=='lstm':
    ner_model = BiLSTM(config.embed_num, config.embed_dim, config.lstm_hiddens,
                    config.lstm_layers,len(config.NER_LABEL_TO_ID), config.dropout)

re_model = SoftmaxNN(sentence_encoder, len(config.RE_LABEL_TO_ID), config.RE_LABEL_TO_ID)    


gcn_model = GCNClassifier(config.lstm_hiddens,
                            len(config.NER_LABEL_TO_ID),
                            encode_hidden_size=100,
                            re_label_num=len(config.RE_LABEL_TO_ID))

if config.use_gcn:
    warmup_step=2
    data_set = AceDataSet(config.train_filename, sentence_encoder)
    val_dataset = AceDataSet(config.test_filename, sentence_encoder)

    val_iter = data.DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            collate_fn=function.collate_fn,num_workers=2)

    train_iter = data.DataLoader(data_set,
                                batch_size=config.batch_size,
                                shuffle=True,
                                collate_fn=function.collate_fn,num_workers=2)
    
    '''
    model_idx='best'
    re_model.load_state_dict(torch.load('./saved_models/combine/re{}/{}'.format(config.load_path,model_idx))['state_dict'])
    ner_model.load_state_dict(torch.load('./saved_models/combine/ner{}/{}'.format(config.load_path,model_idx))['state_dict'])
    gcn_model.load_state_dict(torch.load('./saved_models/combine/gcn{}/{}'.format(config.load_path,model_idx))['state_dict'])
    '''

else:
    data_set = SimpleAceDataSet(config.train_filename, sentence_encoder)
    val_dataset = SimpleAceDataSet(config.test_filename, sentence_encoder)

    val_iter = data.DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            collate_fn=function.simple_collate_fn,num_workers=2)

    train_iter = data.DataLoader(data_set,
                                batch_size=config.batch_size,
                                shuffle=True,
                                collate_fn=function.simple_collate_fn,num_workers=2)



trainer = Trainer(ner_model,
                    re_model,
                    gcn_model,
                    config.use_gcn,
                    train_iter,
                    val_iter,
                    config.path,
                    config.ner_model,
                    config.re_model,
                    config.nerweight,
                    config.reweight,
                    batch_size=config.batch_size,
                    warmup_step=warmup_step,
                    f1_score=config.f1_score)
#model_idx='best'
#re_model.load_state_dict(torch.load('./saved_models/re061622-sig-pcnn-wiki/{}'.format(model_idx))['state_dict'])
#trainer.train_model()
trainer.eval_model(1)

