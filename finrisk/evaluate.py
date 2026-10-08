"""Out-of-sample classification/regression/interval evaluation."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score,mean_absolute_error,mean_squared_error
from .conformal import mc_predict,base_interval,conformity_scores,weighted_quantile,calibrated_interval

def ece(probs,y,n_bins=15):
    confidence=probs.max(1);pred=probs.argmax(1);result=0.
    for lo,hi in zip(np.linspace(0,1,n_bins+1)[:-1],np.linspace(0,1,n_bins+1)[1:]):
        mask=(confidence>=lo)&(confidence<hi if hi<1 else confidence<=hi)
        if mask.any():result+=mask.mean()*abs((pred[mask]==y[mask]).mean()-confidence[mask].mean())
    return float(result)

def collect(model,loader,device,mc=0):
    result={k:[] for k in ('prob','mean','std','true','label','time')}
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            if mc:
                mean,std,prob=mc_predict(model,batch,mc)
            else:
                model.eval();o=model(batch);mean=o['distress'];std=torch.zeros_like(mean);prob=o['logits'].softmax(-1)
        for key,value in zip(result,(prob,mean,std,batch['distress'],batch['labels'],batch['timestamps'])):
            result[key].append(value.detach().cpu().numpy())
    return {key:np.concatenate(value) for key,value in result.items()}

def score_classification(x):
    y=x['label'];p=x['prob'];classes=p.shape[1]
    metrics={'accuracy':float(accuracy_score(y,p.argmax(1))),
             'macro_f1':float(f1_score(y,p.argmax(1),average='macro',zero_division=0)),
             'ece':ece(p,y)}
    for c in range(classes):
        target=(y==c).astype(int)
        metrics[f'pr_auc_class_{c}']=float(average_precision_score(target,p[:,c])) if target.sum() else None
        metrics[f'roc_auc_class_{c}']=float(roc_auc_score(target,p[:,c])) if 0<target.sum()<len(target) else None
    metrics['macro_auroc']=float(np.mean([v for k,v in metrics.items() if k.startswith('roc_auc_class_') and v is not None]))
    metrics['mae']=float(mean_absolute_error(x['true'],x['mean']))
    metrics['rmse']=float(np.sqrt(mean_squared_error(x['true'],x['mean'])))
    return metrics

def interval_metrics(cal,test,alpha=.1,decay=.005):
    cl,cu=base_interval(torch.tensor(cal['mean']),torch.tensor(cal['std']))
    residual=conformity_scores(torch.tensor(cal['true']),cl,cu).numpy()
    q=weighted_quantile(residual,cal['time'],alpha,decay)
    tl,tu=base_interval(torch.tensor(test['mean']),torch.tensor(test['std']))
    low,high=calibrated_interval(tl,tu,q)
    low=low.numpy();high=high.numpy()
    return {'alpha':alpha,'conformal_q':q if np.isfinite(q) else None,'empirical_coverage':float(np.mean((test['true']>=low)&(test['true']<=high))),
            'mean_interval_width':float((high-low).mean())}
