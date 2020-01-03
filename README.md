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
缺点：1.把输入X的所有信息有压缩到一个固定长度的隐向量Z，忽略了输入输入X的长度，当输入句子长度很长，特别是比训练集中最初的句子长度还长时，模型的性能急剧下降。
      2.把输入X编码成一个固定的长度，对于句子中每个词都赋予相同的权重，这样做是不合理的，比如，在机器翻译里，输入的句子与输出句子之间，往往是输入一个或几个词对应于输出的一个或几个词。因此，对输入的每个词赋予相同权重，这样做没有区分度，往往是模型性能下降。
理解：https://zhuanlan.zhihu.com/p/27608348
-------------------------------------------------------------------
第三部分：seq2seq+attention
由于传统的seq2seq是把所有输入信息压入定长C中，encoder的每个输入所占的贡献默认是一致的，这个是不合理的
因此引入soft attention
此模型与传统seq2seq不同的是，再decoder中增加一个context，用context和dec_output共同决定输出
context=attn_weights*enc_output
attn_weights=enc_output(做一个线性变化)*dec_output
参考理解：https://zhuanlan.zhihu.com/p/40920384
