import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2=nn.Parameter(torch.zeros(features))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2


class SubLayerConnection(nn.Module):
    '''
    A residual connection followed by a Layer Norm
    '''
    def __init__(self,size,dropout):
        super(SubLayerConnection, self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,sublayer):
        '''
        Apply residual connection to any sublayer with the same size
        :param x:
        :param sublayer:
        :return:
        '''
        return x+self.dropout(sublayer(self.norm(x)))


