"""Train and select model by validation loss; test set never used for selection."""
import argparse,json,random,math
from pathlib import Path
import numpy as np
import torch,yaml
from finrisk.data import make_loaders
from finrisk.model import FinLLMRisk
from finrisk.losses import objective

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='config.yaml');ap.add_argument('--out',default='outputs');args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config,encoding='utf-8'))
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders,weights=make_loaders(cfg);weights=weights.to(device)
    model=FinLLMRisk(cfg['data'],cfg['model']).to(device);params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('device:',device,'trainable parameters:',params)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    steps_per_epoch=len(loaders[0]); total=steps_per_epoch*cfg['training']['epochs']
    warmup=min(cfg['training']['warmup_steps'],max(1,total//4))
    def factor(step):
        if step<warmup:return max(.01,(step+1)/warmup)
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    best=float('inf');stale=0;out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    history=[]
    for epoch in range(cfg['training']['epochs']):
        totals=[]
        for phase,loader in [('train',loaders[0]),('validation',loaders[1])]:
            model.train(phase=='train');losses=[]
            for batch in loader:
                batch={k:v.to(device) for k,v in batch.items()}
                with torch.set_grad_enabled(phase=='train'):
                    output=model(batch);loss=objective(output,batch,weights,cfg['training'])
                    if phase=='train':
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step()
                losses.append(float(loss.detach().cpu()))
            totals.append(float(np.mean(losses)))
        history.append({'epoch':epoch+1,'train_loss':totals[0],'validation_loss':totals[1]})
        print(f'epoch {epoch+1:03d}: train={totals[0]:.4f} val={totals[1]:.4f}',flush=True)
        if totals[1]<best-1e-5:
            best=totals[1];stale=0
            torch.save({'state_dict':model.state_dict(),'config':cfg,'epoch':epoch+1,'best_val':best},out/'best.pt')
        else:
            stale+=1
            if stale>=cfg['training']['patience']:break
    (out/'history.json').write_text(json.dumps(history,indent=2))
    print('Checkpoint:',out/'best.pt')

if __name__=='__main__':main()
