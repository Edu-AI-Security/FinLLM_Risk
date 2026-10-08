"""MME branches; optional graph attention for node neighborhoods."""
import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,d,p):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Dropout(p),nn.Linear(d,d))
        self.norm=nn.LayerNorm(d)
    def forward(self,x): return self.norm(x+self.net(x))

class StructuredEncoder(nn.Module):
    def __init__(self,inp,d,p):
        super().__init__(); self.bn=nn.BatchNorm1d(inp)
        self.proj=nn.Linear(inp,d)
        self.blocks=nn.Sequential(*(ResidualBlock(d,p) for _ in range(4)))
    def forward(self,x):
        b,t,f=x.shape
        x=self.bn(x.reshape(-1,f)).reshape(b,t,f)
        return self.blocks(self.proj(x))

class GraphNeighborhoodAttention(nn.Module):
    """Three-layer 8-head local GAT-like feature aggregator with neighbor masks.
    Real edge-index GAT requires entity graph snapshots. This consumes
    precomputed neighborhood features [B,T,N,D] (including own node at index 0).
    """
    def __init__(self,inp,d,p):
        super().__init__(); self.proj=nn.Linear(inp,d)
        self.attn=nn.ModuleList([nn.MultiheadAttention(d,8,dropout=p,batch_first=True) for _ in range(3)])
        self.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
    def forward(self,x):
        if x.ndim==3: return self.proj(x)  # cached node embeddings, no graph structure provided
        b,t,n,f=x.shape
        h=self.proj(x).reshape(b*t,n,-1)
        for layer,norm in zip(self.attn,self.norm):
            q=h[:,:1]; updated,_=layer(q,h,h,need_weights=False)
            h=norm(h+updated.expand_as(h))
        return h[:,0].reshape(b,t,-1)

class MultiModalEncoder(nn.Module):
    def __init__(self,ds,dt,dg,d,p):
        super().__init__()
        self.structured=StructuredEncoder(ds,d,p)
        # Document embeddings are precomputed by a frozen or LoRA fine-tuned backbone.
        self.text=nn.Sequential(nn.Linear(dt,d),nn.LayerNorm(d))
        self.graph=GraphNeighborhoodAttention(dg,d,p)
        self.mixer=nn.Sequential(nn.Linear(d*3,d),nn.GELU(),nn.LayerNorm(d))
    def forward(self,structured,text,graph):
        a=self.structured(structured); b=self.text(text); c=self.graph(graph)
        return (a,b,c),self.mixer(torch.cat([a,b,c],dim=-1))
