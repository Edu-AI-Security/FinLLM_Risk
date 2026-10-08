"""Six-layer hierarchical short/long attention with feature-wise gating."""
from torch import nn
import torch

class HTALayer(nn.Module):
    def __init__(self,d,hs,hl,p):
        super().__init__()
        self.pre=nn.LayerNorm(d)
        self.short=nn.MultiheadAttention(d,hs,dropout=p,batch_first=True)
        self.long=nn.MultiheadAttention(d,hl,dropout=p,batch_first=True)
        self.gate=nn.Linear(d*2,d)
        self.post=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(p),nn.Linear(2*d,d))
    def forward(self,query,short,long):
        q=self.pre(query); s=self.pre(short); l=self.pre(long)
        h_s,_=self.short(q,s,s,need_weights=False)
        h_l,_=self.long(q,l,l,need_weights=False)
        gate=torch.sigmoid(self.gate(torch.cat([h_s,h_l],dim=-1)))
        y=query+gate*h_s+(1-gate)*h_l
        return y+self.ff(self.post(y)),gate

class HierarchicalTemporalAttention(nn.Module):
    def __init__(self,d=512,w=20,stride=60,layers=6,hs=8,hl=4,p=.1):
        super().__init__(); self.w=w; self.stride=stride
        self.layers=nn.ModuleList([HTALayer(d,hs,hl,p) for _ in range(layers)])
    def forward(self,seq):
        short=seq[:,-min(self.w,seq.size(1)):]
        # Anchor long-range samples to latest day: t,t-S,t-2S...
        index=torch.arange(seq.size(1)-1,-1,-self.stride,device=seq.device).flip(0)
        long=seq.index_select(1,index)
        q=seq[:,-1:]
        for layer in self.layers: q,_=layer(q,short,long)
        return q.squeeze(1)
