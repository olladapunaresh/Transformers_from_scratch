import math

import torch
import torch.nn.functional as  F
from Encoder import clones
import torch.nn as nn

def attention(query,key,value,mask=None,dropout=None):
    "Compute Scaled dot product Attention"

    d_k=query.size(-1)
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scores=scores.masked_fill(mask==0,-1e9)
    p_attn=F.softmax(scores,dim=-1)
    if dropout is not None:
        p_attn=dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        "Take in number of heads and the model size"

        super(MultiHeadedAttention,self).__init__()
        self.d_k=d_model//h
        self.h=h
        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask=mask.unsqeeze(1)
        nbatches=query.size(0)

        #1. do all the linear projections in batch from d_model => h X d_k

        query,key,value = \
                [l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
                 for l,x in zip(self.linears,(query,key,value))]

        #2. Apply attention on the all the projected vectors in batch

        x,self.attn=attention(query,key,value,mask=mask,dropout=self.dropout)

        x=x.transpose(1,2).contiguous() \
            .view(nbatches,-1,self.h*self.d_k)

        return self.linears[-1](x)

