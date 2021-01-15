import json
import numpy as np
import torch
dataset='wiki' #ace、wiki

use_gcn=True
gcn_layers=3
path='test'
load_path='061719-gcn-cnn-layer3-wiki-1-1'
batch_size=1

re_model='cnn'  #pcnn、cnn、bert、att_bilstm
ner_model='lstm'

f1_score='both' #marco、micro、both
if dataset=='ace':
    NER_LABEL_TO_ID={'padding': 0, 'O': 1, 'GPE': 2, 'WEA': 3, 'PER': 4, 'ORG': 5, 'VEH': 6, 'LOC': 7, 'FAC': 8}
    RE_LABEL_TO_ID={"None": 0, "ART": 1, "ORG": 2, "PAR": 3, "GEN": 4, "PHY": 5, "PER": 6}
    id2ner={0: 'padding', 1 :'O',2: 'GPE', 3: 'WEA', 4: 'PER',5:  'ORG', 6: 'VEH',7:  'LOC', 8: 'FAC'}
    id2re={0: "None", 1: "ART", 2: "ORG", 3: "PAR",4:  "GEN", 5: "PHY", 6: "PER"}
    nerweight=torch.tensor([0,1.46981487, 23.17366947, 128.17839752 , 5.49586228,20.76705157,103.82036572,88.98432698,56.46548362])
    reweight=torch.tensor([2.1189817376867737, 19.436548223350254, 6.548097477554511, 12.636963696369637, 17.28668171557562, 8.256603773584906, 15.194444444444445])
    
elif dataset=='wiki':
    NER_LABEL_TO_ID={'padding': 0, 'O': 1, 'Thing': 2, 'PopulatedPlace': 3, 'Person': 4, 'Organisation': 5, 'Settlement': 6, 'Film': 7, 'Activity': 8, 'NaturalPlace': 9, 'Work': 10, 'MusicalWork': 11, 'Place': 12, 'Building': 13, 'University': 14, 'Company': 15, 'Software': 16, 'Olympics': 17, 'MusicalArtist': 18}
    RE_LABEL_TO_ID={"None":0,'P0': 1, 'B0': 2, 'P17': 3, 'B17': 4, 'P641': 5, 'B641': 6, 'P131': 7, 'B131': 8, 'P47': 9, 'B47': 10, 'P31': 11, 'B31': 12, 'P27': 13, 'B27': 14, 'P118': 15, 'B118': 16, 'P161': 17, 'B161': 18, 'P54': 19, 'B54': 20, 'P361': 21, 'B361': 22, 'P577': 23, 'B577': 24, 'P19': 25, 'B19': 26, 'P106': 27, 'B106': 28, 'P530': 29, 'B530': 30, 'P136': 31, 'B136': 32, 'P279': 33, 'B279': 34, 'P175': 35, 'B175': 36, 'P30': 37, 'B30': 38, 'P155': 39, 'B155': 40}
    id2ner={0: 'padding', 1: 'O', 2: 'Thing', 3: 'PopulatedPlace', 4: 'Person', 5: 'Organisation', 6: 'Settlement', 7: 'Film', 8: 'Activity', 9: 'NaturalPlace', 10: 'Work', 11: 'MusicalWork', 12: 'Place', 13: 'Building', 14: 'University', 15: 'Company', 16: 'Software', 17: 'Olympics', 18: 'MusicalArtist'}
    id2re={0: "None",1: 'P0', 2: 'B0', 3: 'P17', 4: 'B17', 5: 'P641', 6: 'B641', 7: 'P131', 8: 'B131', 9: 'P47', 10: 'B47', 11: 'P31', 12: 'B31', 13: 'P27', 14: 'B27', 15: 'P118', 16: 'B118', 17: 'P161', 18: 'B161', 19: 'P54', 20: 'B54', 21: 'P361', 22: 'B361', 23: 'P577', 24: 'B577', 25: 'P19', 26: 'B19', 27: 'P106', 28: 'B106', 29: 'P530', 30: 'B530', 31: 'P136', 32: 'B136', 33: 'P279', 34: 'B279', 35: 'P175', 36: 'B175', 37: 'P30', 38: 'B30', 39: 'P155', 40: 'B155'}

    #nerweight=torch.tensor([0, 0.01, 0.02748559067908649, 0.05683271290955074, 0.06303580433686333, 0.07822277847309136, 0.3202767190852897, 0.44249745563963006, 0.4644250417982538, 0.6395088571976721, 0.6433764395547835, 0.8461668641056016, 1.1074197120708749, 1.1684973124561815, 1.6636167027116953, 1.7316017316017316, 1.9168104274487252, 2.416626389560174, 2.443195699975568])
    #reweight=torch.tensor([0,0.06262564269565817, 0.06262564269565817, 0.14596622341590157, 0.14596622341590157, 0.20948111527745772, 0.20948111527745772, 0.25408440683995226, 0.25408440683995226, 0.3088803088803089, 0.3088803088803089, 0.37328754339467696, 0.37328754339467696, 0.5482456140350878, 0.5482456140350878, 0.7915149596327371, 0.7915149596327371, 0.9816432708353784, 0.9816432708353784, 1.0067451927917044, 1.0067451927917044, 1.0751532093323297, 1.0751532093323297, 1.0918222513374822, 1.0918222513374822, 1.3248542660307367, 1.3248542660307367, 1.4710208884966165, 1.4710208884966165, 1.649620587264929, 1.649620587264929, 1.6592002654720426, 1.6592002654720426, 1.7467248908296942, 1.7467248908296942, 1.8083182640144666, 1.8083182640144666, 1.8508236165093466, 1.8508236165093466, 2.487562189054726, 2.487562189054726])

    nerweight=None
    reweight=None




vocab_file='./dataset/{}/vocab.pkl'.format(dataset)

word2id = json.load(open('./dataset/{}/word2id.json'.format(dataset)))
word2vec = np.load('./dataset/{}/embedding.npy'.format(dataset))
train_filename='./dataset/{}/train.json'.format(dataset)
test_filename='./dataset/{}/test.json'.format(dataset)


padding_idx=0
embed_num=len(word2vec)
embed_dim=100
lstm_hiddens=50
lstm_layers=1
dropout=0.5
label_paddingId=0
cuda=True
optim='sgd'

is_train=True

pretrained_embed=True
# GCN model
PAD_ID=0
gcn_embedding_dim=100
gcn_hiddens=50


