"""Example conversion of already aligned CSV rows into a model-ready NPZ.
CSV embedding columns are serialized JSON vectors; one row = enterprise observation
at one trading day. Production preprocessing must separately perform point-in-time
filing joins, evidence retrieval, graph snapshot construction, and future labels.
"""
import argparse,json
import pandas as pd
import numpy as np

def vectors(frame,col):
    return np.stack(frame[col].map(lambda v:np.asarray(json.loads(v),dtype=np.float32)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--out',required=True)
    p.add_argument('--length',type=int,default=250);args=p.parse_args()
    df=pd.read_csv(args.csv).sort_values(['company_id','date'])
    # Data must already contain leakage-free features and labels.
    entries=[]
    for _,group in df.groupby('company_id',sort=False):
        group=group.reset_index(drop=True)
        if len(group)<args.length:continue
        for j in range(args.length-1,len(group)):
            window=group.iloc[j-args.length+1:j+1]
            row=group.iloc[j]
            entries.append((vectors(window,'structured'),vectors(window,'text'),vectors(window,'graph'),
                            vectors(window,'knowledge'),np.asarray(json.loads(row['rationale']),dtype=np.float32),
                            int(row['label']),float(row['distress']),pd.Timestamp(row['date']).value/86400e9))
    if not entries:raise RuntimeError('No valid windows')
    keys=('structured','text','graph','knowledge','rationale','labels','distress','timestamps')
    np.savez_compressed(args.out,**{key:np.asarray([e[k] for e in entries]) for k,key in enumerate(keys)})
    print('Saved',len(entries),'windows to',args.out)
if __name__=='__main__':main()
