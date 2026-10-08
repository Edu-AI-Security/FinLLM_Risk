"""Load saved model, calibrate on held-out split, evaluate untouched test split."""
import argparse,json
from pathlib import Path
import torch
from finrisk.data import make_loaders
from finrisk.model import FinLLMRisk
from finrisk.evaluate import collect,score_classification,interval_metrics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='outputs/best.pt')
    parser.add_argument('--out',default='outputs/metrics.json');parser.add_argument('--mc',type=int,default=None)
    args=parser.parse_args()
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=ckpt['config'];loaders,_=make_loaders(cfg)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=FinLLMRisk(cfg['data'],cfg['model']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    mc=cfg['calibration']['mc_samples'] if args.mc is None else args.mc
    cal=collect(model,loaders[2],device,mc=mc);test=collect(model,loaders[3],device,mc=mc)
    metrics=score_classification(test)
    metrics.update(interval_metrics(cal,test,cfg['calibration']['alpha'],cfg['calibration']['decay']))
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(metrics,indent=2,allow_nan=False))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
