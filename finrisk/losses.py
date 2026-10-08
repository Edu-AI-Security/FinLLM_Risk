import torch
import torch.nn.functional as F

def soft_ece(logits,labels,bins=15,temperature=.05):
    prob=logits.softmax(-1); confidence,pred=prob.max(-1)
    # Differentiable approximation to correctness via probability mass on true class.
    correctness=prob.gather(1,labels[:,None]).squeeze(1)
    centers=torch.linspace(0,1,bins,device=logits.device)
    membership=torch.softmax(-((confidence[:,None]-centers[None,:])/temperature)**2,dim=-1)
    mass=membership.sum(0).clamp_min(1e-8)
    avg_conf=(membership*confidence[:,None]).sum(0)/mass
    avg_correct=(membership*correctness[:,None]).sum(0)/mass
    return ((mass/mass.sum())*(avg_conf-avg_correct).square()).sum()

def objective(out,batch,class_weights,train_cfg):
    cls=F.cross_entropy(out['logits'],batch['labels'].long(),weight=class_weights)
    reg=F.huber_loss(out['distress'],batch['distress'])
    cot=(1-F.cosine_similarity(out['reasoning'],out['target_reasoning'],dim=-1)).mean()
    cal=soft_ece(out['logits'],batch['labels'].long())
    return cls+train_cfg['lambda_reg']*reg+train_cfg['lambda_cot']*cot+train_cfg['lambda_cal']*cal
