import numpy as np
import torch
import torch.nn as nn
from Encoder import clones
from LayerNorm import LayerNorm,SubLayerConnection

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,memory,src_mask,tgt_mask):
        '''
        Pass the input and the mask through each layer
        '''
        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    '''
    Decoder is made of self attn ,src attn and feed forward
    '''
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.size=size
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SubLayerConnection(size,dropout),3)

    def forward(self,x,memory,src_mask,tgt_mask):
        m=memory
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x=self.sublayer[1](x,lambda x:self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)


def subsequent_mask(size):
    '''
    Mask out subsequent positions
    :param size:
    :return:
    '''
    attn_shape=(1,size,size)
    subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)==0

