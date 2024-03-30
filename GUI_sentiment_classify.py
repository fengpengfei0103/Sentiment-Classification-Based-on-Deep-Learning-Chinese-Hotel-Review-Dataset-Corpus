#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   GUI_sentiment_classify.py
@Time    :   2024/03/30 13:10:41
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
import tkinter as tk
from tkinter import messagebox

class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx["<unk>"]

    @classmethod
    def build(cls, data, min_freq=1, reserved_tokens=None, stop_words='hit_stopwords.txt'):
        token_freqs = defaultdict(int)
        stopwords = open(stop_words).read().split('\n')
        for i in tqdm(range(data.shape[0]), desc=f"Building vocab"):
            for token in jieba.lcut(data.iloc[i]["review"]):
                if token in stopwords:
                    continue
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx_to_token[index] for index in ids]

def build_data(data_path:str):
    whole_data = pd.read_csv(data_path)
    vocab = Vocab.build(whole_data)
    train_data = [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in whole_data[whole_data["label"] == 1][:2000]["review"]]\
    +[(vocab.convert_tokens_to_ids(sentence), 0) for sentence in whole_data[whole_data["label"] == 0][:2000]["review"]]

    test_data = [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in whole_data[whole_data["label"] == 1][2000:]["review"]]\
        +[(vocab.convert_tokens_to_ids(sentence), 0) for sentence in whole_data[whole_data["label"] == 0][2000:]["review"]]
    return train_data, test_data, vocab

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        embeds = self.embedding(inputs)
        x_pack = pack_padded_sequence(embeds, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        outputs = self.output(hn[-1])
        log_probs = F.log_softmax(outputs, dim=-1)
        return log_probs

def train_model(train_data_loader,vocab):
    # train_data, test_data, vocab = build_data("jiudian_senti_100kUTF8.csv")
    # train_dataset = MyDataset(train_data)
    # test_dataset = MyDataset(test_data)
    # batch_size = 1024
    # train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    # test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class)
    model.to(device)

    nll_loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
            inputs, lengths, targets = [x.to(device) for x in batch]
            log_probs = model(inputs, lengths)
            loss = nll_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epoch}, Loss: {total_loss:.2f}")
    messagebox.showinfo("Training Complete", "Model training completed successfully!")
    return model


def ftest_model(test_data_loader,model):
    acc = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc="Testing"):
            inputs, lengths, targets = [x.to(device) for x in batch]
            output = model(inputs, lengths)
            acc += (output.argmax(dim=1) == targets).sum().item()
    accuracy = acc / len(test_data_loader)
    messagebox.showinfo("Testing Complete", f"Model accuracy on test data: {accuracy:.2%}")

def start_training():
    train_data, test_data, vocab = build_data("jiudian_senti_100kUTF8.csv")
    train_dataset = MyDataset(train_data)
    test_dataset = MyDataset(test_data)
    batch_size = 1024
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    model = train_model(train_data_loader,vocab)
    ftest_model(test_data_loader,model)

def main():
    global embedding_dim, hidden_dim, batch_size, num_epoch, num_class

    # GUI setup
    root = tk.Tk()
    root.title("Sentiment Analysis using LSTM")

    tk.Label(root, text="Sentiment Analysis using LSTM").pack(pady=10)
    # tk.Button(root, text="Start Training", command=start_training).pack(pady=5)

    tk.Label(root, text="Parameters:").pack()
    tk.Label(root, text="Embedding Dimension:").pack()
    embedding_dim_entry = tk.Entry(root)
    embedding_dim_entry.insert(0, "128")
    embedding_dim_entry.pack()
    tk.Label(root, text="Hidden Dimension:").pack()
    hidden_dim_entry = tk.Entry(root)
    hidden_dim_entry.insert(0, "24")
    hidden_dim_entry.pack()
    tk.Label(root, text="Batch Size:").pack()
    batch_size_entry = tk.Entry(root)
    batch_size_entry.insert(0, "1024")
    batch_size_entry.pack()
    tk.Label(root, text="Number of Epochs:").pack()
    num_epoch_entry = tk.Entry(root)
    num_epoch_entry.insert(0, "20")
    num_epoch_entry.pack()
    tk.Label(root, text="Number of Classes:").pack()
    num_class_entry = tk.Entry(root)
    num_class_entry.insert(0, "2")
    num_class_entry.pack()

    # Function to start training with custom parameters
    def start_custom_training():
        global embedding_dim, hidden_dim, batch_size, num_epoch, num_class
        embedding_dim = int(embedding_dim_entry.get())
        hidden_dim = int(hidden_dim_entry.get())
        batch_size = int(batch_size_entry.get())
        num_epoch = int(num_epoch_entry.get())
        num_class = int(num_class_entry.get())
        start_training()

    tk.Button(root, text="Start Training with Custom Parameters", command=start_custom_training).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
