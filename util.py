"""
此部分包含多个函数的封装
    pre_data()  去空格，取单词，得到单词词典
    random_batch()  随机批量抽取
"""

import numpy as np


# 去空格，取单词，得到单词词典
def pre_data(sentences):
    word_sequence = ' '.join(sentences).split()
    word_list = ' '.join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    return word_sequence, word_list, word_dict





if __name__ == '__main__':
    pass
