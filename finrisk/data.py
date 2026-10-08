"""Data formats, chronological splitting and synthetic smoke-test fixture."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RiskDataset(Dataset):
    """NPZ arrays: structured[N,T,128], text[N,T,D], graph[N,T,G],
    knowledge[N,T,K], rationale[N,K], labels[N], distress[N], timestamps[N].
    All document/knowledge/graph vectors must be generated using ONLY dated evidence.
    Each row represents a prediction origin; target labels are strictly future horizon.
    """
    def __init__(self, arrays, indices, mean=None, std=None):
        self.arrays = arrays
        self.indices = np.asarray(indices)
        self.mean, self.std = mean, std
    def __len__(self): return len(self.indices)
    def __getitem__(self, index):
        i = self.indices[index]
        fields = ('structured','text','graph','knowledge','rationale','labels','distress','timestamps')
        out = {key: torch.as_tensor(self.arrays[key][i]) for key in fields}
        if self.mean is not None:
            out['structured'] = (out['structured'].float() - self.mean) / self.std
        for k in ('text','graph','knowledge','rationale','distress','timestamps'):
            out[k] = out[k].float()
        return out

def synthetic_data(n=320, t=250, ds=128, dt=256, dg=64, dk=256, seed=42):
    """Artificial inputs exclusively for verifying that the pipeline runs."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n,1,1)).astype('float32')
    structured = (rng.standard_normal((n,t,ds)).astype('float32') + .4*z)
    text = (rng.standard_normal((n,t,dt)).astype('float32') + .2*z)
    graph = rng.standard_normal((n,t,dg)).astype('float32')
    knowledge = rng.standard_normal((n,t,dk)).astype('float32')
    logits = 0.7*structured[:,-1,0] + 0.4*structured[:,-1,1] + z[:,0,0]
    cutoffs = np.quantile(logits, [.40,.68,.84,.94])
    labels = np.searchsorted(cutoffs, logits).astype('int64')
    distress = np.clip(0.12 + .16*labels + rng.normal(0,.07,n),0,1).astype('float32')
    rationale = knowledge[:,-1].copy()
    # increasing timestamps for reproducible chronological splits
    times = np.arange(n,dtype='float32')
    return dict(structured=structured,text=text,graph=graph,knowledge=knowledge,
                rationale=rationale,labels=labels,distress=distress,timestamps=times)

def load_arrays(cfg):
    d=cfg['data']
    if d['source']=='synthetic':
        return synthetic_data(d['num_samples'],d['sequence_length'],d['structured_dim'],
                              d['text_dim'],d['graph_dim'],d['knowledge_dim'],cfg['seed'])
    if d['source']!='npz' or not d['path']:
        raise ValueError('Set data.source to synthetic or npz and supply data.path.')
    with np.load(Path(d['path']),allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    required={'structured','text','graph','knowledge','rationale','labels','distress','timestamps'}
    if missing:=required-set(arrays): raise ValueError(f'Missing NPZ arrays: {sorted(missing)}')
    return arrays

def make_loaders(cfg):
    arrays=load_arrays(cfg)
    order=np.argsort(arrays['timestamps'],kind='stable'); n=len(order)
    d=cfg['data']; a=int(n*d['train_fraction']); b=a+int(n*d['val_fraction']); c=b+int(n*d['calibration_fraction'])
    if not 0<a<b<c<n: raise ValueError('Need nonempty train/val/cal/test splits')
    # NOTE: rows from same entity with overlapping label horizons must be purged upstream.
    train_idx=order[:a]
    mean=arrays['structured'][train_idx].mean(axis=(0,1)).astype('float32')
    std=arrays['structured'][train_idx].std(axis=(0,1)).astype('float32').clip(1e-6)
    stats=(torch.from_numpy(mean),torch.from_numpy(std))
    sets=[RiskDataset(arrays,ix,*stats) for ix in (train_idx,order[a:b],order[b:c],order[c:])]
    bs=cfg['training']['batch_size']
    loaders=[DataLoader(ds,batch_size=bs,shuffle=(i==0),num_workers=0) for i,ds in enumerate(sets)]
    counts=np.bincount(arrays['labels'][train_idx].astype(int),minlength=d['num_classes']).clip(1)
    weights=np.sum(counts)/(d['num_classes']*counts)
    return loaders,torch.tensor(weights,dtype=torch.float32)
