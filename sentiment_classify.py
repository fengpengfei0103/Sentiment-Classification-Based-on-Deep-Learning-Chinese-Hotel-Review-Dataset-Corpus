#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sentiment_classify.py
@Time    :   2024/01/02 14:40:41
@Author  :   Pengfei F
@Version :   1.0
@Contact :   fpf0103@163.com
@Desc    :   None
'''

# ‘’‘此处使用中文酒店评论情感分析数据集，基于LSTM模型实现情感分析’‘’

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import jieba
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from torch import optim
from torch.nn import functional as F

'''定义一个词表类型。'''
# 该类用于实现token到索引的映射
class Vocab:

    def __init__(self, tokens = None) -> None:
        # 构造函数
        # tokens：全部的token列表

        self.idx_to_token = list()
        # 将token存成列表，索引直接查找对应的token即可
        self.token_to_idx = dict()
        # 将索引到token的映射关系存成字典，键为索引，值为对应的token

        if tokens is not None:
            # 构造时输入了token的列表
            if "<unk>" not in tokens:
                # 不存在标记
                tokens = tokens + "<unk>"
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
                # 当前该token对应的索引是当下列表的最后一个
            self.unk = self.token_to_idx["<unk>"]

    @classmethod
    def build(cls, data, min_freq=1, reserved_tokens=None, stop_words='hit_stopwords.txt'):
        # 构建词表
        # cls：该类本身
        # data: 输入的文本数据
        # min_freq：列入token的最小频率
        # reserved_tokens：额外的标记token
        # stop_words：停用词文件路径
        token_freqs = defaultdict(int)
        stopwords = open(stop_words).read().split('\n')
        for i in tqdm(range(data.shape[0]), desc=f"Building vocab"):
            for token in jieba.lcut(data.iloc[i]["review"]):
                if token in stop_words:
                    continue
                token_freqs[token] += 1
        # 统计各个token的频率
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        # 加入额外的token
        uniq_tokens += [token for token, freq in token_freqs.items() \
            if freq >= min_freq and token != "<unk>"]
        # 全部的token列表
        return cls(uniq_tokens)

    def __len__(self):
        # 返回词表的大小
        return len(self.idx_to_token)

    def __getitem__(self, token):
        # 查找输入token对应的索引，不存在则返回<unk>返回的索引
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        # 查找一系列输入标签对应的索引值
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        # 查找一系列索引值对应的标记
        return [self.idx_to_token[index] for index in ids]

'''数据集构建函数'''
def build_data(data_path:str):
    '''
    Args:
       data_path:待读取本地数据的路径 
    Returns:
       训练集、测试集、词表
    '''
    whole_data = pd.read_csv(data_path)
    # 读取数据为 DataFrame 类型
    vocab = Vocab.build(whole_data)
    # 构建词表

    train_data = [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in whole_data[whole_data["label"] == 1][:2000]["review"]]\
    +[(vocab.convert_tokens_to_ids(sentence), 0) for sentence in whole_data[whole_data["label"] == 0][:2000]["review"]]
    # 分别取褒贬各2000句作为训练数据，将token映射为对应的索引值

    test_data = [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in whole_data[whole_data["label"] == 1][2000:]["review"]]\
        +[(vocab.convert_tokens_to_ids(sentence), 0) for sentence in whole_data[whole_data["label"] == 0][2000:]["review"]]
    # 其余数据作为测试数据

    return train_data, test_data, vocab

'''声明一个 DataSet 类'''
class MyDataset(Dataset):

    def __init__(self, data) -> None:
        # data：使用词表映射之后的数据
        self.data = data

    def __len__(self):
        # 返回样例的数目
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
'''声明一个collate_fn函数，用于对一个批次的样本进行整理'''
def collate_fn(examples):
    # 从独立样本集合中构建各批次的输入输出
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    # 获取每个序列的长度
    inputs = [torch.tensor(ex[0]) for ex in examples]
    # 将输入inputs定义为一个张量的列表，每一个张量为句子对应的索引值序列
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 目标targets为该批次所有样例输出结果构成的张量
    inputs = pad_sequence(inputs, batch_first=True)
    # 将用pad_sequence对批次类的样本进行补齐
    return inputs, lengths, targets

'''创建一个LSTM类作为模型'''
class LSTM(nn.Module):
    # 基类为nn.Module
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        # 构造函数
        # vocab_size:词表大小
        # embedding_dim：词向量维度
        # hidden_dim：隐藏层维度
        # num_class:多分类个数
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 词向量层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        # lstm层
        self.output = nn.Linear(hidden_dim, num_class)
        # 输出层，线性变换

    def forward(self, inputs, lengths):
        # 前向计算函数
        # inputs:输入
        # lengths:打包的序列长度
        # print(f"输入为：{inputs.size()}")
        embeds = self.embedding(inputs)
        # 注意这儿是词向量层，不是词袋词向量层
        # print(f"词向量层输出为：{embeds.size()}")
        x_pack = pack_padded_sequence(embeds, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        # LSTM需要定长序列，使用该函数将变长序列打包
        # print(f"经过打包为：{x_pack.size()}")
        hidden, (hn, cn) = self.lstm(x_pack)
        # print(f"经过lstm计算后为：{hn.size()}")
        outputs = self.output(hn[-1])
        # print(f"输出层输出为：{outputs.size()}")
        log_probs = F.log_softmax(outputs, dim = -1)
        # print(f"输出概率值为：{probs}")
        # 归一化为概率值
        return log_probs

'''训练'''
# 超参数设置
embedding_dim = 128
hidden_dim = 24
batch_size = 1024
num_epoch = 20
num_class = 2

train_data, test_data, vocab = build_data("jiudian_senti_100kUTF8.csv")
# 加载数据
train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class)
model.to(device)
# 加载模型

nll_loss = nn.NLLLoss()
# 负对数似然损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Adam优化器

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        # print(inputs.size())
        log_probs = model(inputs, lengths)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss:{total_loss:.2f}")

# 测试
acc = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, lengths, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, lengths)
        acc += (output.argmax(dim=1) == targets).sum().item()
print(f"ACC:{acc / len(test_data_loader):.2f}")
