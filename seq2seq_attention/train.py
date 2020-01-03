from seq2seq_attention.seq2seq_atten import *
from util import *

import torch
import torch.nn as nn

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps


def train(max_epoch,lr,input_batch,hidden,output_batch,target_batch,model):
    optimizer=torch.optim.Adam(model.parameters(),lr)
    criterion=nn.CrossEntropyLoss()
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        output,_=model.forward(input_batch,hidden,output_batch)
        loss = criterion(output, target_batch.squeeze(0).long())
        if (epoch+1)%200==0:
            print('Epoch:{0},loss:{1}'.format(epoch,loss))
        loss.backward()
        optimizer.step()

def test():
    test_batch=[np.eye(n_class)[[word_dict[n] for n in 'SPPPP']]]
    test_batch=Variable(torch.Tensor(test_batch))
    predict,train_attn=model(input_batch,hidden,test_batch)
    predict=predict.data.max(1,keepdim=True)[1]
    print(sentences[0],'-->',[num_dict[n.item()] for n in predict.squeeze()])


if __name__=="__main__":
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    word_sequence, word_list, word_dict, n_class, num_dict = pre_data(sentences)
    # 模型参数
    n_hidden=64
    dropout=0.1
    model=Attention(n_class,n_hidden,dropout)
    model.init_parameters()
    # 开始训练
    input_batch,output_batch,target_batch=make_batch(sentences,n_class,word_dict)
    max_epoch=1000
    lr=0.01
    hidden=Variable(torch.zeros(1,1,n_hidden))
    train(max_epoch,lr,input_batch,hidden,output_batch,target_batch,model)
    #测试
    test()
