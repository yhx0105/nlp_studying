"""
此部分包含多个函数的封装
    pre_data()  去空格，取单词，得到单词词典
    random_batch()  随机批量抽取
    make_batch()
"""

import numpy as np
from torch.autograd import Variable
import torch

# 去空格，取单词，得到单词词典
def pre_data(sentences):
    word_sequence = ' '.join(sentences).split()
    word_list = ' '.join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    # 适用于seq2seq_attn
    num_dict = {i: w for i, w in enumerate(word_list)}
    return word_sequence, word_list, word_dict, n_class, num_dict


# for seq2seq_attention 假设数据集已经做好了标记
def make_batch(sentences, n_class, word_dict):
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]
    return Variable(torch.Tensor(input_batch)),Variable(torch.Tensor(output_batch)),Variable(torch.Tensor(target_batch))


if __name__ == '__main__':
    # sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    # word_sequence, word_list, word_dict, n_class, num_dict = pre_data(sentences)
    # input_batch,output_batch,target_batch=make_batch(sentences,n_class,word_dict)
    # print(target_batch)
    pass
