import pandas as pd
import warnings
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import pickle as pkl
import tqdm
import operator
from importlib import import_module
import sys
import argparse
from torch.nn import DataParallel
from dataset_iter import DatasetIterdtor
sys.path.append('.')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def atten_score_process(func_to_cwe, batch_id, atten_score, predict, true, vul_idx, mode, p_line_num=2):
    p_true = 0.0
    p_false = 0.0
    f_line = 0.0
    tp_true = 0.0
    tp_false = 0.0
    code_data = pd.read_pickle('./data/big_vul/' + mode +  '_seq.pkl')
    id_to_code = {}
    '''将函数id和code保存为字典结构'''
    for item in code_data:
        id_to_code[int(item[0].split('@')[-1])] = item[1]

    top_10 = ['CWE-119', 'CWE-20', 'CWE-125', 'CWE-399', 'CWE-264', 'CWE-200', 'CWE-190', 'CWE-476', 'CWE-787', 'CWE-416'] #'CWE-189'
    '''每个batch的data格式为(layer_num, batch_size, num_heads, sequence_length, sequence_length)'''
    for pos in range(len(batch_id)):
        atten_scores = np.zeros(512)
        # layer_data = source_data[pos][1]  #取一个sequence, atten_score格式为[id, atten_score]
        id = batch_id[pos]
        # print(id)
        line_to_idx = {}
        st = 1
        line_num = 0
        for i, cd in enumerate(id_to_code[id]):
            if cd == '</s>':
                line_to_idx[line_num] = [st, i]
            elif cd == 'Ċ':
                line_to_idx[line_num] = [st, i]
                st = i + 1
                line_num += 1
        for i in range(12):
            layer_data = atten_score[i][pos]
            head_data = np.zeros(512)
            for idx in range(12):
                token_score =[]
                for j in range(512):
                    token_score.append(layer_data[idx][j][j])
                head_data  = head_data + np.array(token_score)
            atten_scores = atten_scores + head_data
    
        line_to_score = {}
        for li, idx in line_to_idx.items():
            if li ==0:
                continue
            st = idx[0]
            end = idx[1]
            sc = 0.0
            for i in range(end-st):
                sc += atten_scores[st+i]
            line_to_score[str(li)] = sc
        final_scores = dict(sorted(line_to_score.items(), key=operator.itemgetter(1), reverse=True))
        tag = False  #行数预测正确则tag为True
        
        if predict[pos] == 0 and true[pos] == 0:
            p_true += 1.0
            
        if len(vul_idx[pos]) == 0:
            continue
        # flag = True
        # for index in vul_idx[pos]:
        #     if int(index) > len(final_scores):
        #         flag = False
        #         break
        # if flag == False:
        #     continue
        # p_line_num = len(vul_idx[pos]) + 15
        # all_in = True
        # for i in range(len(vul_idx[pos])):
        #     if vul_idx[pos][i] not in list(final_scores.keys())[:p_line_num]:
        #         all_in = False
        # if all_in == True:
        #     print(id, func_to_cwe[id])
        #     print(vul_idx[pos])
        #     print(list(final_scores.keys())[:p_line_num])
        for ln in list(final_scores.keys())[:p_line_num]:
        # for ln in list(final_scores.keys()):
            if str(ln) in vul_idx[pos]:
                tag = True
        if tag == True:
            if func_to_cwe[id] in top_10:
                # print(func_to_cwe[id])
                tp_true += 1.0
            p_true += 1.0
        else:
            if func_to_cwe[id] in top_10:
                tp_false += 1.0
            p_false += 1.0
        # print(predict[pos])
        # print(type(predict[pos]))
        # print(true[pos])
        # print(type(true[pos]))
        # print(id)
        # print(final_scores)
        # print(vul_idx[pos])
    return p_true, p_false, tp_true, tp_false

def evaluate_atten_score(config, model, test_iter):
    '''获取文件对应的CWE信息'''
    func_to_cwe = {}
    with open('big_vul_msg.txt', 'r') as fp:
        cwe_msg = fp.read().split('\n--------------------------\n')
        for msg in cwe_msg:
            # print(msg.split('&&'))
            if msg.split('&&') == ['']:
                continue
            f_name = msg.split('&&')[0].split('@')[-1].split('.')[0]
            cwe_id = msg.split('&&')[2]
            func_to_cwe[int(f_name)] = cwe_id

    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    true_p = 0.0
    false_p = 0.0
    true_tp = 0.0
    false_tp = 0.0
    with torch.no_grad():
        for id, s, x1, x2, label, vul_idx in tqdm.tqdm(test_iter):
            s = s.cuda()
            x1 = x1.cuda()
            x2 = x2.cuda()
            label = label.cuda()
            outs, atten_score = model(s, x1, x2)
            atten_score = np.array([score.cpu().detach().numpy() for score in atten_score])
            predict = torch.max(outs.data, 1)[1].cpu().numpy()
            true =label.data.cpu().numpy()
            # print(type(predict))
            # print(predict)
            id = id.data.cpu().numpy()
            p_t, p_f, tp_t, tp_f = atten_score_process(func_to_cwe, id, atten_score, predict, true, vul_idx, 'test', p_line_num=17)#20,24,26,28,30,34,  38,40
            true_p += p_t
            false_p += p_f
            true_tp += tp_t
            false_tp += tp_f

    line_p = true_p / (true_p + false_p)
    line_tp = true_tp / (true_tp + false_tp)
    print("prediction result of vulnerable line...")
    print(line_p)
    print(line_tp)

def test(config, model, test_iter, test=True):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    test_acc, test_loss, test_precision, test_recall, test_f1, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    
    msg = 'Test Acc: {0:>6.2%}, Test precision: {1:>6.2%}, Test recall: {2:>6.2%}, Test f1: {3:>6.2%}'
    print(msg.format(test_acc, test_precision, test_recall, test_f1))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)

def evaluate(config, model, val_iter, test=False):
    model.eval()
    total_losses = 0
    total_labels = np.array([], dtype=int)
    total_predicts = np.array([], dtype=int)
    with torch.no_grad():
        for id, s, x1, x2, label, vul_idx in tqdm.tqdm(val_iter):
            # id = id.cuda()
            s = s.cuda()
            x1 = x1.cuda()
            x2 = x2.cuda()
            label = label.cuda()
            outs, _ = model(s, x1, x2)
            loss = F.cross_entropy(outs, label)
            total_losses += loss
            label = label.data.cpu().numpy()
            predict = torch.max(outs.data, 1)[1].cpu().numpy()
            total_labels = np.append(total_labels, label)
            total_predicts = np.append(total_predicts, predict)

    acc = metrics.accuracy_score(total_labels, total_predicts)
    if test:
        precision = metrics.precision_score(total_labels, total_predicts)
        recall = metrics.recall_score(total_labels, total_predicts)
        f1 = metrics.f1_score(total_labels, total_predicts)
        report = metrics.classification_report(total_labels, total_predicts, target_names=['0','1'],digits=3)
        confusion = metrics.confusion_matrix(total_labels, total_predicts)
        return acc, total_losses / len(val_iter), precision, recall, f1, report, confusion
    return acc, total_losses / len(val_iter)


def train(config, model, train_iter, val_iter, test_iter):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    dev_best_acc = float(0)
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step()
        n_batch = 0
        for id, s, x1, x2, label, vul_idx in tqdm.tqdm(train_iter):
            model.zero_grad()
            # id = id.cuda()
            s = s.cuda()
            x1 = x1.cuda()
            x2 = x2.cuda()
            label = label.cuda()
            outs, _ = model(s, x1, x2)
            # model.zero_grad()
            batch_losses = F.cross_entropy(outs, label)
            batch_losses.backward()
            optimizer.step()
            n_batch += 1
        if n_batch % len(train_iter) == 0:
        # if epoch % 1 == 0:  # best at 360
            true = label.data.cpu()
            predict = torch.max(outs.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(true, predict)
            dev_acc, dev_loss = evaluate(config, model, val_iter)
            if dev_acc > dev_best_acc:
                dev_best_loss = dev_loss
                torch.save(model.module.state_dict(), config.save_path)
                
            msg = 'Epoch: {0:>1},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
            # msg > 表示在两者之间增加空格
            print(msg.format((epoch+1), batch_losses.item(), train_acc, dev_loss, dev_acc))

            model.train()
    test(config, model.module, test_iter, test=True)
    evaluate_atten_score(config, model.module, test_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='./data/FFmpeg/train_set_token.pkl', help='input train dataset')
    parser.add_argument('--val_data', type=str, default='./data/FFmpeg/val_set_token.pkl', help='input valid dataset')
    parser.add_argument('--test_data', type=str, default='./data/FFmpeg/test_set_token.pkl', help='input test dataset')
    parser.add_argument('--model', type=str, default='CodeBert_Blstm', help='model name')
    # parser.add_argument('--vocab_path', type=str, default='FFmpeg_code_vocab.json', help='vocab path')
    parser.add_argument('--p', type=str, default='FFmpeg', help='input project name')
    args = parser.parse_args()

    model_name = args.model
    X = import_module('module.' + model_name)
    config = X.Config()
    print('[+] loading vocabulary...')
    code_vocab = json.load(open(args.p + '_code_vocab.json'))
    device = config.device
    config.n_vocab = len(code_vocab)
    dir_name = './save_dict/' + args.p
    config.save_path = dir_name + '/' + model_name + '.ckpt'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    args.train_data = './data/' + args.p + '/train_set_token.pkl'
    args.val_data = './data/' + args.p + '/val_set_token.pkl'
    args.test_data = './data/' + args.p + '/test_set_token.pkl'
    train_dataset = pkl.load(open(args.train_data, 'rb'))
    val_dataset = pkl.load(open(args.val_data, 'rb'))
    test_dataset = pkl.load(open(args.test_data, 'rb'))
    train_iter = DatasetIterdtor(train_dataset, config.batch_size, device)
    val_iter = DatasetIterdtor(val_dataset, config.batch_size, device)
    test_iter = DatasetIterdtor(test_dataset, config.batch_size, device)
    model = X.CodeBert_Blstm(config)
    model = model.to(device)
    # device_ids = [0, 1]
    model = DataParallel(model)
    train(config, model, train_iter, val_iter, test_iter)
    model.load_state_dict(torch.load('./save_dict/big_vul/CodeBert_Blstm.ckpt'))
    # acc, test_loss, p, r, f1, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    # msg = 'Test acc: {0:>6.2%}, Test precision: {1:>6.2%}, Test recall: {2:>6.2%}, Test f1: {3:>6.2%}'
    # print(msg.format(acc, p, r, f1))
    # print("Precision, Recall and F1-Score...")
    # print(test_report)
    # print("Confusion Matrix...")
    # print(test_confusion)
    # print("prediction result of vulnerable line...")
    # evaluate_atten_score(config, model, test_iter)