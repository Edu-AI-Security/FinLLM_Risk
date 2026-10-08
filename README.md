# FinLLM-Risk: modular Python reproduction

This is an **executable architectural reproduction** of the supplied FinLLM-Risk manuscript, not an independently validated reproduction of the claimed AUROC 0.871 or 89.4M trainable-parameter, 17B-LLM, 28.3 ms latency findings. The runnable default uses *synthetic data* and precomputed text/knowledge representations. It is **not** the paper's full LLaMA-4 Scout 17B LoRA training run.

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

## Scope and deviations

1. LLaMA-4 Scout 17B, rank-32 / alpha-64 LoRA, max 4096 tokens and NF4 quantization are specified in the paper; **not loaded or trained in this runnable embedding-first implementation**. Encode SEC texts/KA-CoT prompts externally with your licensed model then feed their vectors, or integrate Hugging Face + PEFT separately.
2. The model's KA-CoT path learns evidence-conditioned embeddings and cosine distillation, **not generated natural-language chains of thought**. `rationale` vectors must be real ground-truth rationale embeddings to replicate this training signal.
3. Graph attention runs on supplied neighborhood tensors. With only `[N,T,Dg]`, the graph branch is a projection of cached node features, not actual cross-company message passing.
4. The manuscript does not define precise base interval formula, target rationale construction pipeline or train/calibration split beyond 8:1:1. We use MC mean ± 1.645 SD, and 70/10/10/10 splits to keep calibration separate. These implementation choices may differ from the authors' code.
5. Time-decay residual weighting is a practical heuristic; formal weighted conformal guarantees require additional assumptions/valid likelihood-ratio weights. The interval code reports *observed* test coverage without asserting universal coverage.
6. Synthetic sample outputs are solely software tests and should **never** be presented as reproduced experimental findings.
