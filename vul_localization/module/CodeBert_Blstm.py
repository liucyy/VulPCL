from turtle import forward
import numpy as np
import math
import json
import torch.nn.functional as F
from unicodedata import bidirectional
import torch
import pickle as pkl
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


class Config:
    '''模型参数配置'''
    def __init__(self):
        self.model_name = 'CodeBert_Blstm'
        self.save_path = './save_dict/' + self.model_name + '.ckpt'  # 保存模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        self.dropout = 0.5  #随机失活
        self.require_improvement = 10000  #训练超过5000效果没提升就结束
        self.num_classes = 2  #类别数\
        # self.output_attentions = True
        self.n_vocab = 0  #词汇数，训练时赋值
        self.num_epochs = 22  #训练次数，50轮收敛
        self.batch_size = 16  #mini_batch大小
        self.pad_size = 512  #每句话处理的长度大小（截取或填补）
        self.learning_rate = 2e-5  #学习率 
        self.cb_embed = 768
        self.embed = 300  #词向量维度，使用了预训练的词向量则维度一致
        self.hidden_size = 256  #LSTM隐藏层
        self.num_layers = 2  #LSTM层数
    

class CodeBert_Blstm(nn.Module):
    def __init__(self, config):
        super(CodeBert_Blstm, self).__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")  # 处理源代码序列
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)  # 处理pdg序列
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers, 
                            bidirectional=True, batch_first=True, dropout=config.dropout)  # 训练pdg和ast、ddg、cdg序列

        # 从lstm中得到输出后，将out输入到以下三个Linear层中得到Q、K、V
        self.W_Q = nn.Linear(config.hidden_size*2, config.hidden_size*2, bias=False)
        self.W_K = nn.Linear(config.hidden_size*2, config.hidden_size*2, bias=False)
        self.W_V = nn.Linear(config.hidden_size*2, config.hidden_size*2, bias=False)

        self.linear0 = nn.Sequential(
                                    nn.Linear(config.cb_embed, config.hidden_size*2),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2, stride=2),
                                    nn.Linear(config.hidden_size, config.hidden_size),
                                    nn.ReLU()
                                    )
        self.linear1 = nn.Sequential(
                                    nn.MaxPool1d(2, stride=2),
                                    nn.Linear(config.hidden_size, config.hidden_size),
                                    nn.ReLU()
                                    )
        self.linear2 = nn.Sequential(
                                    # nn.MaxPool1d(2, stride=2),
                                    nn.Linear(config.hidden_size*3, 128),
                                    nn.ReLU(),
                                    # nn.MaxPool1d(2, stride=2),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, config.num_classes))
    
    # 加入自注意力机制
    def sf_attention(self, input):
        Q = self.W_Q(input)
        K = self.W_K(input)
        V = self.W_V(input)
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(1,2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        code_context = torch.matmul(alpha_n, V)

        output = code_context.sum(1)
        return output, alpha_n

    def forward(self, s, x1, x2):
        s_code = s
        codebert_out = self.codebert(s_code, output_attentions=True)
        out0 = codebert_out[0]
        out0 = out0[:, 0, :]
        atten_score = codebert_out[2]  #(layer_num, batch_size, num_heads, sequence_length, sequence_length)
        # atten_score = np.array([score.cpu().detach().numpy() for score in atten_score])
        # atten_score = atten_score.swapaxes(0,1)  #(batch_size, layer_num, num_heads, sequence_length, sequence_length)
        # a_score = []
        # for i in range(16):
        #     a_score.append([id[i], atten_score[i]])
        # pkl.dump(a_score, open('atten_score_example.pkl', 'wb'))
        # print(len(atten_score))
        # print(len(atten_score[0]))
        # print(atten_score[0])
        out0 = self.linear0(out0)

        code1 = x1  # [batch_size,seq_len]
        out1 = self.embedding(code1)  # [batch_size,seq_len,embedding_size]
        # print(out1.shape)
        out1, _ = self.lstm(out1)  # [batch_size,selq_len,hidden_size*2]
        # print(out1.shape)
        out1 = torch.cat([out1[:,0,-256:], out1[:,-1,:256]], 1)
        out1 = out1.view(out1.shape[0], -1, out1.shape[1])
        sf_atten_out1, _ = self.sf_attention(out1)
        # print(sf_score[9][9])
        sf_atten_out1 = self.linear1(sf_atten_out1)

        code2 = x2
        out2, _ =self.lstm(code2)
        out2 = torch.cat([out2[:,0,-256:], out2[:,-1,:256]], 1)
        out2 = out2.view(out2.shape[0], -1, out2.shape[1])
        sf_atten_out2, _ = self.sf_attention(out2)
        sf_atten_out2 = self.linear1(sf_atten_out2)

        out = torch.cat([out0, sf_atten_out1, sf_atten_out2], 1)

        # out = self.linear2(out0)

        return out, atten_score
