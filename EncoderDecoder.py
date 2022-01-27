import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.generator=generator
    def forward(self,src,tgt,src_mask,tgt_mask):
        return self.decode(self.encode(src,src_mask),tgt,tgt_mask)

    def encode(self,src,src_mask):
        self.encoder(self.src_embed(src),src_mask)

    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)


class Generator(nn.Module):
    '''
    Define standard linear + softmax generation step
    '''
    def __init__(self,d_model,vocab):
        super(Generator, self).__init__()
        self.proj=nn.Linear(d_model,vocab)
    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)




