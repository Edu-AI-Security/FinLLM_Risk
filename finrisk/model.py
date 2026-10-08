"""Full supervised FinLLM-Risk pipeline, starting from pretrained embeddings."""
import torch
from torch import nn
from .encoders import MultiModalEncoder
from .hta import HierarchicalTemporalAttention

class KnowledgeAugmentedReasoner(nn.Module):
    def __init__(self,k,d,p):
        super().__init__(); self.project=nn.Sequential(nn.Linear(k,d),nn.GELU(),nn.LayerNorm(d))
        self.combine=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(p),nn.LayerNorm(d))
    def forward(self,hta,kb):
        # User must provide evidence retrieved as of prediction time.
        evidence=self.project(kb[:,-1]);return self.combine(torch.cat((hta,evidence),-1))

class FinLLMRisk(nn.Module):
    def __init__(self,data_cfg,model_cfg):
        super().__init__(); d=model_cfg['hidden_dim']; p=model_cfg['dropout']
        self.mme=MultiModalEncoder(data_cfg['structured_dim'],data_cfg['text_dim'],data_cfg['graph_dim'],d,p)
        self.hta=HierarchicalTemporalAttention(d,model_cfg['short_window'],model_cfg['long_stride'],
                        model_cfg['num_layers'],model_cfg['short_heads'],model_cfg['long_heads'],p)
        self.reasoner=KnowledgeAugmentedReasoner(data_cfg['knowledge_dim'],d,p)
        self.target_reasoning=nn.Linear(data_cfg['knowledge_dim'],d,bias=False)
        self.fuse=nn.Sequential(nn.Linear(5*d,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(p))
        self.classifier=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(p),nn.Linear(d//2,data_cfg['num_classes']))
        self.regressor=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(p),nn.Linear(d//2,1),nn.Sigmoid())
    def forward(self,batch):
        mods,sequence=self.mme(batch['structured'],batch['text'],batch['graph'])
        h=self.hta(sequence); r=self.reasoner(h,batch['knowledge'])
        fused=self.fuse(torch.cat([h,*(x[:,-1] for x in mods),r],dim=-1))
        target=self.target_reasoning(batch['rationale'])
        return dict(logits=self.classifier(fused),distress=self.regressor(fused).squeeze(-1),
                    reasoning=r,target_reasoning=target,features=fused)
