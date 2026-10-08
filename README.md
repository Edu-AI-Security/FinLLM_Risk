
## Components

- `finrisk/data.py`: chronological train/validation/calibration/test split, train-only standardization, synthetic fixture, NPZ input
- `finrisk/encoders.py`: 4-block residual MLP, text embedding projection, 3-layer attention aggregation for supplied neighborhoods
- `finrisk/hta.py`: 6-layer short-/long-horizon attention, dynamic gating, W=20, S=60, heads 8/4
- `finrisk/retrieval.py`: timestamp-constrained dense retrieval (simple NumPy FAISS substitute)
- `finrisk/model.py`: multimodal fusion, evidence-conditioned reasoning, 5-class classifier and distress regression
- `finrisk/losses.py`: balanced CE, Huber, cosine distillation and soft-ECE
- `finrisk/conformal.py`: MC dropout with frozen BatchNorm, exponential weighted quantile, calibrated intervals
- `finrisk/evaluate.py`: macro OvR AUROC, F1, accuracy, per-class PR-AUC, ECE, MAE/RMSE, empirical coverage
- `train.py`, `test.py`: end-to-end run
- `prepare_npz.py`: window builder for point-in-time aligned embeddings
- `tests/test_pipeline.py`: smoke tests

## Setup

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### CPU smoke run

Use the supplied `config_smoke.yaml` (small model and synthetic samples):

```bash
python train.py --config config_smoke.yaml --out outputs_smoke
python test.py --checkpoint outputs_smoke/best.pt --out outputs_smoke/metrics.json --mc 5
```

### Paper-scale architecture (compute-intensive)

```bash
python train.py --config config.yaml --out outputs
python test.py --checkpoint outputs/best.pt --out outputs/metrics.json
```

`config.yaml` preserves paper values (d=512, 6 HTA layers, W=20, S=60, 80 epochs, early stopping 10, 50 MC samples, AdamW 2e-4). However the document encoder here accepts precomputed vectors **instead of implementing end-to-end 17B LoRA**. Paper-scale config may take considerable GPU time.

## Real data contract

Prepare arrays in an `.npz` (with `allow_pickle=False`) with these keys:

| Key | Shape | Meaning |
| --- | --- | --- |
| `structured` | N,T,128 | 64 ratio + 64 market features |
| `text` | N,T,Dt | Precomputed document vectors, evidence disclosed by current date |
| `graph` | N,T,Dg or N,T,neighbors,Dg | Precomputed node vectors or local node neighborhoods |
| `knowledge` | N,T,Dk | Precomputed retrieval-augmented prompt/evidence vectors |
| `rationale` | N,Dk | Training-only reference rationale embedding |
| `labels` | N | 0..4 next-horizon risk label |
| `distress` | N | Future target in [0,1] |
| `timestamps` | N | Chronological prediction origins (numeric) |

Choose `data.source: npz`, `data.path: path/to/your_file.npz`, and set all feature dimensions to match arrays in YAML. All modalities must respect disclosure dates: no future `rdq` dates, undisclosed graph links or post-cutoff retrieved passages. The manuscript describes five rule-derived risk levels; actual label construction requires SEC/Compustat corporate event history and should be done upstream. **Purging/embargo for overlapping firm-horizon windows and cross-company event contamination is the dataset producer's responsibility**; simple row-order chronology does not prevent every form of leakage.

The manuscript reports three *independently evaluated* benchmarks. Train a separate run for each supplied dataset, do not concatenate them silently.


