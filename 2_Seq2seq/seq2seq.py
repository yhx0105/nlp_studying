import numpy as np

from torch.autograd import Variable
import torch
import torch.nn as nn


# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
# 处理输入，输出数据
def make_data(seq_data, n_step, num_dic, n_class):
    """
    :param seq_data:需要训练的data
    :param n_step: 设置RNN的步数
    :param num_dic: 字母字典
    :param n_class: 加入S,E,P后字母表长度
    :return: input_batch onthot; output_batch onehot ;target
    """
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))

        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target)

    return Variable(torch.Tensor(input_batch)), Variable(torch.Tensor(output_batch)), Variable(
        torch.Tensor(target_batch))


# model
class Seq2Seq(nn.Module):
    def __init__(self, n_class, n_hidden, dropout, enc_input, enc_hidden, dec_input):
        super(Seq2Seq, self).__init__()
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.enc_input = enc_input
        self.enc_hidden = enc_hidden
        self.dec_input = dec_input

    def produce(self):
        self.enc_cell = nn.RNN(input_size=self.n_class, hidden_size=self.n_hidden, dropout=self.dropout)
        self.dec_cell = nn.RNN(input_size=self.n_class, hidden_size=self.n_hidden, dropout=self.dropout)
        self.fc = nn.Linear(self.n_hidden, self.n_class)

    def forward(self):
        enc_input = self.enc_input.transpose(0, 1)  # enc_input: [max_len(=n_step, time step), batch_size, n_class]
        dec_input = self.dec_input.transpose(0, 1)  # dec_input: [max_len(=n_step, time step), batch_size, n_class]

        # enc_states : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, enc_states = self.enc_cell(enc_input, self.enc_hidden)
        # outputs : [max_len+1(=6), batch_size, num_directions(=1) * n_hidden(=128)]
        outputs, _ = self.dec_cell(dec_input, enc_states)

        model = self.fc(outputs)  # model : [max_len+1(=6), batch_size, n_class]
        return model


def train(model, lr, target_b):
    criteration = nn.CrossEntropyLoss()
    model.produce()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(2000):
        optimizer.zero_grad()
        # input_batch : [batch_size, max_len(=n_step, time step), n_class]
        # output_batch : [batch_size, max_len+1(=n_step, time step) (becase of 'S' or 'E'), n_class]
        # target_batch : [batch_size, max_len+1(=n_step, time step)], not one-hot
        output = model.forward()
        # output : [max_len+1, batch_size, num_directions(=1) * n_hidden]
        output = output.transpose(0, 1)  # [batch_size, max_len+1(=6), num_directions(=1) * n_hidden]
        loss = 0
        for i in range(0, len(target_b)):
            # output[i] : [max_len+1, num_directions(=1) * n_hidden, target_batch[i] : max_len+1]
            loss += criteration(output[i], target_b[i].long())
        if (epoch + 1) % 1000 == 0:
            print('Epoch:{0}\ncost={1}'.format(epoch, loss))
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    num_dic = {n: i for i, n in enumerate(char_arr)}

    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'],
                ['high', 'low']]

    n_class = len(num_dic)
    n_step = 5
    n_hidden = 128
    dropout = 0.4
    batch_size = len(seq_data)
    # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
    enc_hidden = Variable(torch.zeros(1, batch_size, n_hidden))

    input_b, output_b, target_b = make_data(seq_data, n_step, num_dic, n_class)
    # print('input_b:{0}\n,output_b:{1}\n,target_b:{2}'.format(input_b, output_b, target_b))
    model = Seq2Seq(n_class, n_hidden, dropout, input_b, enc_hidden, output_b)
    train(model, 0.01, target_b)



