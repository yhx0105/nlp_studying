nlp_studying
记录自己学习nlp的过程


--------------------------------------------------------------------
第一部分：word_2_vector
word_2_vector 包含CBOW和skip-gram两个部分
skip-gram:输入中心词预测其周围的词，通过构建一个只用一个隐层的神经网络，找到预测概率最大的那个词
参考：理解 https://zhuanlan.zhihu.com/p/27234078
-------------------------------------------------------------------
第二部分：seq2seq
seq2seq:包括编码器和解码器，最直观的理解是分别用两个RNN
其中Encoder将输入编码为固定大小状态向量，把最后一层的RNN的状态为decoder的输出
Decoder读取这个输出，再加上自己的的输入，进行训练。
target用作算loss
理解：https://zhuanlan.zhihu.com/p/27608348

