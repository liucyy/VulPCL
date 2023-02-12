from turtle import forward
import numpy as np
from unicodedata import bidirectional
import torch
import torch.nn as nn


class Config:
    '''模型参数配置'''
    def __init__(self):
        self.model_name = 'composite_BLSTM'
        self.save_path = './save_dict/' + self.model_name + '.ckpt'  # 保存模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        self.dropout = 0.5  #随机失活
        self.require_improvement = 10000  #训练超过5000效果没提升就结束
        self.num_classes = 2  #类别数
        self.n_vocab = 0  #词汇数，训练时赋值
        self.num_epochs = 30  #训练次数，50轮收敛
        self.batch_size = 128  #mini_batch大小
        self.pad_size = 512  #每句话处理的长度大小（截取或填补）
        self.learning_rate = 0.001  #学习率
        self.embed = 300  #词向量维度，使用了预训练的词向量则维度一致
        self.hidden_size = 256  #LSTM隐藏层
        self.num_layers = 2  #LSTM层数
    

class BLSTM_Model(nn.Module):
    def __init__(self, config):
        super(BLSTM_Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers, 
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.linear1 = nn.Sequential(
                                    nn.MaxPool1d(2, stride=2),
                                    nn.Linear(config.hidden_size * 2, config.hidden_size),
                                    nn.ReLU()
                                    )
        self.linear2 = nn.Sequential(
                                    # nn.MaxPool1d(2, stride=2),
                                    nn.Linear(config.hidden_size * 2, 128),
                                    nn.ReLU(),
                                    # nn.MaxPool1d(2, stride=2),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, config.num_classes))
    
    def forward(self, x1, x2):
        code1 = x1  # [batch_size,seq_len]
        out1 = self.embedding(code1)  # [batch_size,seq_len,embedding_size]
        out1, _ = self.lstm(out1)  # [batch_size,selq_len,hidden_size*2]
        out1 = torch.cat([out1[:,-1,:], out1[:,0,:]], 1)
        out1 = self.linear1(out1)
        code2 = x2
        out2, _ =self.lstm(code2)
        out2 = torch.cat([out2[:,-1,:], out2[:,0,:]], 1)
        out2 = self.linear1(out2)

        out = torch.cat([out1,out2],1)
        out = self.linear2(out)
        # out = torch.cat([out1, code2], 1)
        # out, _ = self.lstm(out)
        # out = torch.cat([out[:,-1,:], out[:,0,:]], 1)
        # out = self.linear1(out)
        # out = self.linear2(out)
        return out
