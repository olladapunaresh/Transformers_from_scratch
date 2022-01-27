# Starting code
import copy
from LayerNorm import LayerNorm
import numpy as np
import torch.nn as nn
from LayerNorm import SubLayerConnection
import torch.nn.functional as F
def clones(module,N):
    '''

    :param module:
    :param N:
    :return: Produce N identical layers
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    '''
    Core Encoder is a stack of N layers
    '''
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,mask):
        '''
        Pass the input x and mask through each layer in turn and finally
        norm layer
        :param x:
        :param mask:
        :return:
        '''
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SubLayerConnection(size,dropout),2)
        self.size=size
    def forward(self,x,mask):
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)
