import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, n_class, n_hidden, dropout):
        super(Attention,self).__init__()
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.dropout = dropout

    def init_parameters(self):
        self.enc_cell = nn.RNN(input_size=self.n_class, hidden_size=self.n_hidden, dropout=self.dropout)
        self.dec_cell = nn.RNN(input_size=self.n_class, hidden_size=self.n_hidden, dropout=self.dropout)

        # Linear for attention
        self.attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.out = nn.Linear(self.n_hidden * 2, self.n_class)

    def forward(self, enc_inputs, hidden, dec_inputs):
        enc_inputs = enc_inputs.transpose(0, 1)  # enc_inputs:[max(n_step,time_step),batch_size,n_class]
        dec_inputs = dec_inputs.transpose(0, 1)  # dec_inputs:[max(n_step,time_step),batch_size,n_class]

        enc_output,enc_hidden=self.enc_cell(enc_inputs,hidden)
        # enc_outputs:[n_step,batch_size,num_directions(=1)*n_hidden],matrix F
        # enc_hidden:[num_layers(=1)*num_direction(=1),batch_size,n_hidden]

        trained_attn=[]
        dec_hidden=enc_hidden
        n_step=len(dec_inputs)
        model=Variable(torch.empty([n_step,1,self.n_class]))

        for i in range(n_step): # each time step
            # dec_output : [n_step(=1),batch_size(=1),num_direction(=1)*n_hidden]
            # hidden : [num_layer(=1)*num_direction(=1),batch_size(=1),n_hidden]
            dec_output,hidden=self.dec_cell(dec_inputs[i].unsqueeze(0),dec_hidden)
            attn_weights=self.get_att_weight(dec_output,enc_output)  # attn_weights : [1, 1, n_step]
            trained_attn.append(attn_weights.squeeze().data.numpy())

            # matrix-matrix product of matrices [b,n,m] x [b,m,p] = [b,n,p]
            context=attn_weights.bmm(enc_output.transpose(0,1))
            dec_output=dec_output.squeeze(0)  # dec_output : [batch_size(=1),num_direction(=1)*n_hidden]
            context=context.squeeze(1)  # [1,num_direction(=1)*n_hidden]
            model[i]=self.out(torch.cat((dec_output,context),1))

        return model.transpose(0,1).squeeze(0),trained_attn

    def get_att_weight(self,dec_output,enc_output):  # get attention weight one 'dec_output' with 'enc_outputs'
        n_step=len(enc_output)
        attn_scores=Variable(torch.zeros(n_step)) # attn_scores : [n_step]

        for i in range(n_step):
            attn_scores[i]=self.get_att_score(dec_output,enc_output[i])

        # Normalize scores to weights in range 0 to 1
        return F.softmax(attn_scores).view(1,1,-1)

    def get_att_score(self,dec_output,enc_output):  # enc_output [batch_size,num_direct]
        score=self.attn(enc_output)  # score:[batch_size,n_hidden]
        return torch.dot(dec_output.view(-1),score.view(-1))  # inner product make scalar value
