from util import *

import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

sentences = ["i like dog", "i like cat", "i like animal",
             "dog cat animal", "apple cat dog like", "dog fish milk like",
             "dog cat eyes like", "i like apple", "apple i hate",
             "apple i movie book music like", "cat dog hate", "cat dog like"]


# 获得skip_grams输入数据
def skip_grams(word_sequence):
    """
    :param word_sequence: 输入整个文档的单词
    :return:返回包含[target,context]内容的列表
    """
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

        for w in context:
            skip_grams.append([target, w])
    return skip_grams


# 随机抽取batch_size个数据
def random_batch(data, size, voc_size):
    """
    :param data: skip_gram 是若干个个[target ,context] 的list
    :param size: batch_size 通过np.random.choice从data中随机抽取batch_size个数据
    :param voc_size:生成onehot编码，每个编码对应一个单词
    :return:返回label,和input
    """
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]])
        random_labels.append(data[i][1])

    return random_inputs, random_labels


# model
class Word2Vec(nn.Module):
    def __init__(self, voc_size, embedding_size, dtype):
        super(Word2Vec, self).__init__()
        self.voc_size = voc_size
        self.embedding_size = embedding_size
        self.dtype = dtype

    def init_parameters(self):
        self.W = nn.Parameter(-2 * torch.rand(self.voc_size, self.embedding_size) + 1).type(self.dtype)
        self.WT = nn.Parameter(-2 * torch.rand(self.embedding_size, self.voc_size) + 1).type(self.dtype)
        return self.W, self.WT

    def forward(self, X):
        # X：[batch_size,voc_size]
        hidden_layer = torch.matmul(X, self.W)  # hidden_layer:[batch_size,embedding_size]
        output_layer = torch.matmul(hidden_layer, self.WT)  # output_layer:[batch_size,voc_size]
        return output_layer

    def predict(self, data):
        result = self.forward(data)
        return result


# 训练
def train(model):
    # 随机初始华
    model.init_parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5000):
        input_batch, target_batch = random_batch(skip_grams, batch_size, voc_size)

        input_batch = Variable(torch.Tensor(input_batch))
        target_batch = Variable(torch.LongTensor(target_batch))

        optimizer.zero_grad()
        output = model.forward(input_batch)

        # output:[batch_size,voc_size],target_batch:[batch_size](LongTensor)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:{0}, cost:{1}'.format(epoch + 1, loss))

        loss.backward()
        optimizer.step()


def draw_plt(model):
    for i, label in enumerate(word_list):
        W, WT = model.init_parameters()
        x, y = float(W[i][0]), float(W[i][1])
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()


if __name__ == '__main__':
    word_sequence, word_list, word_dict = pre_data(sentences)
    skip_grams = skip_grams(word_sequence)
    batch_size = 20
    voc_size = len(word_sequence)
    embedding_size = 2
    dtype = torch.FloatTensor
    random_inputs, random_labels = random_batch(skip_grams, batch_size, voc_size)
    data = torch.Tensor(random_inputs)
    # print('random_inputs:{0}\nrandom_labels:{1}'.format(random_inputs,random_labels))
    word2vec = Word2Vec(voc_size, embedding_size, dtype)
    train(word2vec)
    draw_plt(word2vec)
    print(word2vec.predict(data).shape)
