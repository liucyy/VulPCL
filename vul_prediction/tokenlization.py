from pkgutil import get_data
import sys
import re
import os
import numpy as np
import pickle as pkl
import pandas as pd
import json
import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from multiprocessing import Pool
import argparse
sys.path.append('.')

from cd_features_extracting import SPECIAL_WORD


def code_padding(tokens, maxlen=512, pos='r-pre', embeddings=None):
    if embeddings == None:
        tokens = [code_vocab.get(t, SPECIAL_WORD['<UNK>']) for t in tokens]
    else:
        tokens = [nodes_vocab.get(t) for t in tokens]
    if len(tokens) > maxlen:
        tokens = tokens[:maxlen]
    if pos == 'pre':
        tokens = [SPECIAL_WORD['<PAD>']] * (maxlen - len(tokens)) + tokens
    else:
        tokens = tokens + [SPECIAL_WORD['<PAD>']] * (maxlen - len(tokens))
    if embeddings != None:
        tokens = [embeddings[str(t)] for t in tokens]
    return tokens

def get_feature_token(data_path, embeddings, tokenizer):
    data = pd.read_pickle(data_path)
    vec_pkl = []
    bar = tqdm.tqdm(total=len(data))
    for _, item in data.iterrows():
        id = item.id
        s_code = item.code
        tokens1 = item.feature1
        tokens2 = item.feature2
        label = item.label
        s_code = re.sub(r'(\s)+', ' ', s_code).strip()  #替换源代码中的空格、换行符等
        s_tokens = ''.join(s_code.split(' '))
        s_tokens = tokenizer.tokenize(s_tokens)[:510]  #截断，只保留tokens长度为512的部分
        s_tokens = [tokenizer.cls_token] + s_tokens + [tokenizer.sep_token]
        s_ids =  tokenizer.convert_tokens_to_ids(s_tokens)
        padding_length = 512 - len(s_ids)
        s_ids += [tokenizer.pad_token_id] * padding_length
        tokens1 = code_padding(tokens1, 512) # token:FFmpeg1024， qemu512 cfgnode:256, 256
        tokens2 = code_padding(tokens2, 256, embeddings=embeddings) 
        vec_pkl.append([id, s_ids, tokens1, tokens2, label])
        bar.update()
    bar.close()
    data_path = data_path.replace('.pkl', '_token.pkl')
    pkl.dump(vec_pkl, open(data_path, 'wb'))

def get_dataset_split(pkl_path, pro):
    db = pd.read_pickle(pkl_path)
    la = []
    for _, item in db.iterrows():
        la.append(item.label)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=6)
    sss.get_n_splits(db, la)
    store_path = './data/' + pro
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    
    train_set_path = store_path + '/train_set.pkl'
    val_test_set_path = store_path + '/val_test_set.pkl'
    for train_idx, val_test_idx in sss.split(db, la):
        train_data = pd.DataFrame({'id':db['id'][train_idx], 'code': db['code'][train_idx], 'feature1':db['feature1'][train_idx], 'feature2':db['feature2'][train_idx], 'label':db['label'][train_idx]})
        train_data.to_pickle(train_set_path)
        val_test_data = pd.DataFrame({'id':db['id'][val_test_idx], 'code': db['code'][val_test_idx], 'feature1':db['feature1'][val_test_idx], 'feature2':db['feature2'][val_test_idx], 'label':db['label'][val_test_idx]})
        val_test_data.to_pickle(val_test_set_path)
    
    tv = pd.read_pickle(val_test_set_path)
    pla = []
    for _, item in tv.iterrows():
        pla.append(item.label)
    sss_pa = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=9)
    sss_pa.get_n_splits(tv, pla)
    val_set_path = store_path + '/val_set.pkl'
    test_set_path = store_path + '/test_set.pkl'
    for val_idx, test_idx in sss_pa.split(tv, pla):
            val_data = pd.DataFrame({'id':db['id'][val_idx], 'code': db['code'][val_idx], 'feature1':db['feature1'][val_idx], 'feature2':db['feature2'][val_idx], 'label':db['label'][val_idx]})
            val_data.to_pickle(val_set_path)
            test_data = pd.DataFrame({'id':db['id'][test_idx], 'code': db['code'][test_idx], 'feature1':db['feature1'][test_idx], 'feature2':db['feature2'][test_idx], 'label':db['label'][test_idx]})
            test_data.to_pickle(test_set_path)

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default='FFmpeg', help='input project name')
    return parser.parse_args()

args = get_parameter()
code_vocab = json.load(open(args.p + '_code_vocab.json'))
nodes_vocab = json.load(open(args.p + '_nodes_vocab.json'))

if __name__ == '__main__':
    '''划分数据集'''
    print('[+] split dataset...')
    get_dataset_split('./data/' + args.p + '_graph_input.pkl', args.p)

    embeddings = pd.read_pickle(args.p + '_embeddings.pkl')
    root_path = './data/' + args.p
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    print('[+] transform tokens to vector...')
    train_data_path = root_path + '/train_set.pkl'
    val_data_path = root_path + '/val_set.pkl'
    get_feature_token(train_data_path, embeddings, tokenizer)
    get_feature_token(val_data_path, embeddings, tokenizer)
    test_data_path = root_path + '/test_set.pkl'
    get_feature_token(test_data_path, embeddings, tokenizer)