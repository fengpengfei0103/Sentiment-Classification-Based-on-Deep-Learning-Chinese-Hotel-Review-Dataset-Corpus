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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading


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


class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis using DeepLearingLSTM")

        self.embedding_dim = tk.IntVar(value=128)
        self.hidden_dim = tk.IntVar(value=24)
        self.batch_size = tk.IntVar(value=1024)
        self.num_epoch = tk.IntVar(value=20)
        self.num_class = tk.IntVar(value=2)

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Sentiment Analysis using DeepLearingLSTM").pack(pady=10)

        tk.Label(self.root, text="Parameters:").pack()
        tk.Label(self.root, text="Embedding Dimension:").pack()
        tk.Entry(self.root, textvariable=self.embedding_dim).pack()
        tk.Label(self.root, text="Hidden Dimension:").pack()
        tk.Entry(self.root, textvariable=self.hidden_dim).pack()
        tk.Label(self.root, text="Batch Size:").pack()
        tk.Entry(self.root, textvariable=self.batch_size).pack()
        tk.Label(self.root, text="Number of Epochs:").pack()
        tk.Entry(self.root, textvariable=self.num_epoch).pack()
        tk.Label(self.root, text="Number of Classes:").pack()
        tk.Entry(self.root, textvariable=self.num_class).pack()

        tk.Button(self.root, text="Start Training with Custom Parameters", command=self.start_custom_training).pack(
            pady=10)

        # 添加用于显示训练过程的图表
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

    def start_custom_training(self):
        threading.Thread(target=self.train).start()

    def train(self):
        embedding_dim = self.embedding_dim.get()
        hidden_dim = self.hidden_dim.get()
        batch_size = self.batch_size.get()
        num_epoch = self.num_epoch.get()
        num_class = self.num_class.get()

        train_data, test_data, vocab = build_data("jiudian_senti_100kUTF8.csv")
        train_dataset = MyDataset(train_data)
        test_dataset = MyDataset(test_data)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
        model = self.train_model(train_data_loader, vocab, num_epoch)

        self.ftest_model(test_data_loader, model)

    def ftest_model(self, test_data_loader, model):
        acc = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for batch in tqdm(test_data_loader, desc="Testing"):
                inputs, lengths, targets = [x.to(device) for x in batch]
                output = model(inputs, lengths)
                acc += (output.argmax(dim=1) == targets).sum().item()
        accuracy = acc / len(test_data_loader)
        messagebox.showinfo("Testing Complete", f"Model accuracy on test data: {accuracy:.2%}")

    def update_plot(self, epoch_losses):
        self.ax.clear()
        self.ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.canvas.draw()

    def train_model(self, train_data_loader, vocab, num_epoch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTM(len(vocab), self.embedding_dim.get(), self.hidden_dim.get(), self.num_class.get())
        model.to(device)

        nll_loss = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        epoch_losses = []
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
            epoch_losses.append(total_loss)
            self.update_plot(epoch_losses)
            print(f"Epoch {epoch + 1}/{num_epoch}, Loss: {total_loss:.2f}")
        messagebox.showinfo("Training Complete", "Model training completed successfully!")
        return model


def main():
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()