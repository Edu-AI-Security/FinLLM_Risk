"""Time-decayed weighted conformal intervals with finite-sample correction.
NOTE: arbitrary time-decay does not by itself guarantee coverage under drift.
"""
import numpy as np
import torch

def weighted_quantile(scores,times,alpha=.1,decay=.005,test_weight=1.):
    scores=np.asarray(scores,float);times=np.asarray(times,float)
    if len(scores)==0:raise ValueError('Empty calibration scores')
    weights=np.exp(-decay*(times.max()-times));weights=weights/weights.max()
    # Add hypothetical +inf test point; if quantile lands there, return infinity.
    order=np.argsort(scores);sorted_weights=weights[order]
    threshold=(1-alpha)*(weights.sum()+test_weight)
    idx=np.searchsorted(np.cumsum(sorted_weights),threshold,side='left')
    return float(scores[order[idx]]) if idx<len(scores) else float('inf')

@torch.no_grad()
def mc_predict(model,batch,n=50):
    # Dropout ON while BatchNorm statistics remain fixed (model.eval first).
    model.eval()
    for mod in model.modules():
        if isinstance(mod,torch.nn.Dropout): mod.train()
    outputs=[model(batch) for _ in range(n)]
    regression=torch.stack([o['distress'] for o in outputs])
    probabilities=torch.stack([o['logits'].softmax(-1) for o in outputs])
    return regression.mean(0),regression.std(0,unbiased=False),probabilities.mean(0)

def base_interval(mean,std,z=1.645):
    return (mean-z*std).clamp(0,1),(mean+z*std).clamp(0,1)

def conformity_scores(truth,lower,upper):
    return torch.maximum(torch.maximum(lower-truth,truth-upper),torch.zeros_like(truth))

def calibrated_interval(lower,upper,q):
    return (lower-q).clamp(0,1),(upper+q).clamp(0,1)
