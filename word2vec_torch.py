import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
from collections import Counter
import random
import pickle


# 训练 Word2Vec 的数据集类
class Word2VecDataset(Dataset):
    def __init__(self, corpus, window_size):
        self.window_size = window_size
        self.data = [word for sentence in corpus for word in sentence]
        self.word_to_idx = {}
        self.data_pairs = []

        # 构建词典
        for word in self.data:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)

        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

        # 保存 index2word
        with open('./data/word2vec/index2word.pkl', 'wb') as f:
            pickle.dump(self.idx_to_word, f)

        # 保存 word2index
        with open('./data/word2vec/word2index.pkl', 'wb') as f:
            pickle.dump(self.word_to_idx, f)

        # 生成训练样本
        for sentence in corpus:
            sentence_ix = [self.word_to_idx[word] for word in sentence]
            for i, center_word in enumerate(sentence_ix):
                context_words = sentence_ix[max(0, i-self.window_size):i] + sentence_ix[i+1:min(i+self.window_size+1, len(sentence_ix))]
                for context_word in context_words:
                    self.data_pairs.append((center_word, context_word))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        center_word, context_word = self.data_pairs[index]
        # return  input, target
        return torch.tensor(center_word, dtype=torch.long), torch.tensor(context_word, dtype=torch.long)


# Word2Vec torch 模型         embedding => linear 选取唯一的预测值
class Word2Vec_torch(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec_torch, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        output = self.linear(embeds)
        return output

    # embedding 保存下来
    def save_embeddings(self, file_path):
        embeddings = self.embedding.weight.data.cpu().numpy()
        np.save(file_path, embeddings)

# Word2Vec torch 模型 训练函数
def train_word2vec(model, dataset, batch_size, num_epochs, learning_rate, save_path):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for i, data in enumerate(dataloader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, dataset.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss/len(dataloader)))
        torch.save(model.state_dict(), save_path)

        # 保存embedding，后续计算相近词
        model.save_embeddings('./data/word2vec/embeddings.npy')


# Word2Vec torch 模型 测试函数
def test_word2vec(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for i in range(101, 150):
        inputs = torch.LongTensor([i])
        inputs.to(device)
        output = model(inputs)
        _, indices = torch.topk(output, 5)
        indices = indices.detach().numpy().squeeze()
        print('Word:', dataset.idx_to_word[i])
        print('Similar Words:', [dataset.idx_to_word[x] for x in indices])
        print('#'*20)


