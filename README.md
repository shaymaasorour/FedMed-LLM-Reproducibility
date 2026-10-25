# FedMed-LLM — code release

Heterogeneous multi-task federated learning for emergency-department analytics: **five independent hospital clients**,
each with a local input adapter and a local task-specific head; only shared representation / adaptation parameters
(shared encoder in Part 1, LoRA adapters in Part 2) take part in federated coordination. Datasets are never merged.

## Repository layout
| Path | Content |
|---|---|
| `configs/default.yaml` | complete experimental configuration (Stages A–D, heterogeneity protocol, seeds); identical to `CFG` in the notebook |
| `fedmed/data_tabular.py` | per-hospital preprocessing (cleaning, temporal features, label / embedding encoding, scaling, split) |
| `fedmed/tabular.py` | local adapter + shared encoder + local head; Fed-GDBD, ZeroDFL (ring gossip, no server), TACTFL (two-stage) |
| `fedmed/llm.py` | federated LLM engine: FedAvg, FedProx, Fed-GDBD, ZeroDFL, TACTFL, centralized reference, Extractor-Only; label-skew protocol |
| `fedmed/systems.py` | Stage C: device population of the five clients and round-by-round run of the federation (Figures 14–15) |
| `fedmed/lora_proxy.py` | Stage D: LoRA hybrid tables (TreeXformer / ClinicalBERT proxy / TFT proxy) on the five clients |
| `fedmed/analysis.py` | Part 3: text statistics, mean ± SD and 95 % CI, paired tests with Holm correction, calibration, class imbalance, non-LLM baselines, edge-hardware cost |
| `scripts/prepare_splits.py` | writes the per-client train/test splits |
| `scripts/run_tabular.py` | one Part 1 experiment |
| `scripts/run_llm.py` | one Part 2 experiment |
| `scripts/run_systems.py` | Figures 14–15 |
| `scripts/run_lora_proxy.py` | LoRA Tables 1–3 and the Table 9 federated-vs-centralized test |
| `scripts/run_repeated.py` | repeated runs over all seeds, methods and backbones (Section 3.2) |
| `scripts/run_statistics.py` | significance, calibration and class-imbalance analyses (Sections 3.3–3.4) |
| `scripts/run_baselines.py` | non-LLM baselines (Section 3.5) |
| `scripts/text_and_edge.py` | free-text statistics (Section 3.1) and edge-hardware cost (Section 3.7) |

## Data availability
All five datasets are publicly available on Kaggle; place them in `./datasets/` (Part 1) and the text exports
`dataset_1.csv … dataset_5.csv` (columns `text`, `label` + structured fields) in the working directory (Part 2).
Every client must be present: a missing file or target column stops the run.

| Client | Dataset | File | Source |
|---|---|---|---|
| 1 | Hospital Emergency Dataset | `hospital-emergency-dataset.csv` | https://www.kaggle.com/datasets/xavierberge/hospital-emergency-dataset |
| 2 | Hospital Triage and Patient History | `hospital-triage-and-patient-history-data.csv` | https://www.kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data |
| 3 | Integrated Emergency Response Dataset (IERAD) | `integrated-emergency-response-dataset-ierad.csv` | https://www.kaggle.com/datasets/datasetengineer/integrated-emergency-response-dataset-ierad |
| 4 | Hospital Patient Data (AV Healthcare Analytics II) | `hospital-patient-data.csv` | https://www.kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii |
| 5 | Healthcare Dataset (synthetic) | `healthcare-dataset.csv` | https://www.kaggle.com/datasets/prasad22/healthcare-dataset |

Processed splits can be regenerated exactly with `scripts/prepare_splits.py` (stratified 80/20 split per client, seed 42;
the split indices are sorted so the arrival order used by TACTFL is preserved).

## Reproducing the experiments
Python 3.8 (pinned versions in `requirements.txt`); the LLM runs need a GPU.
```bash
pip install -r requirements.txt
python scripts/prepare_splits.py --data-dir ./datasets --out ./splits --seed 42
python scripts/run_tabular.py --method fedgdbd --data-dir ./datasets          # Part 1 (Table 4 settings)
python scripts/run_llm.py --backbone GLM-RoBERTa --method tactfl --seed 11 \
       --out results/glm_tactfl_s11.csv                                        # Part 2 (Section 4.4 settings)
python scripts/run_llm.py --backbone GLM-RoBERTa --method tactfl --seed 11 --label-skew-alpha 0.3 \
       --train-fraction 0.4                                                    # heterogeneity sweep point
python scripts/run_llm.py --backbone GLM-RoBERTa --method tactfl --seed 11 --tau 0.5   # tau sensitivity
```
Seeds used for all repeated runs: **11, 22, 33, 44, 55**.

## Mapping to the paper
| Paper item | Configuration | Command |
|---|---|---|
| Table 4, Part 1 models | Stage A | `run_tabular.py --method ...` |
| Section 4.4, Tables 5–9, Figure 10 | Stage B | `run_llm.py --backbone ... --method ...` |
| FedAvg / FedProx baselines | Stage B | `run_llm.py --method fedavg` / `--method fedprox` |
| Figures 10–12, EMD table | heterogeneity | `run_llm.py --label-skew-alpha A --train-fraction 0.4` |
| New Results (low-resource protocol) | Stage B | `run_llm.py --train-fraction 0.3` |
| Per-task heterogeneity figure, Extractor-Only | heterogeneity | `run_llm.py --method extractor --label-skew-alpha A --train-fraction 0.4` |
| Input ablation | Stage B | `run_llm.py --input-mode structured|text|both` |
| TACTFL sensitivity (Section 3.6b) | Stage B | `run_llm.py --method tactfl --tau T` / `--label-fraction F` |
| Figures 14–15 | Stage C | `run_systems.py --backbone T5-Large` |
| LoRA Tables 1–3 | Stage D | `run_lora_proxy.py --table 1|2|3` (add `--new-results` for the New Results tables) |
| Table 9 federated vs centralized (Section 3.3b) | Stage D | `run_lora_proxy.py --table9` |
| Repeated runs, mean ± SD, 95 % CI (Section 3.2) | Stage B | `run_repeated.py` |
| Significance, calibration, class imbalance (Sections 3.3–3.4) | Stage B | `run_statistics.py --imbalance` |
| Non-LLM baselines (Section 3.5) | Stage A | `run_baselines.py` |
| Free-text statistics, edge-hardware cost (Sections 3.1, 3.7) | Stage B | `text_and_edge.py` |

The notebook contains the same code together with the figures.

## Method details that matter for reproduction
- **Fed-GDBD:** sample-weighted averaging of the shared parameters; class prototypes are turned into global
  decision-boundary prototypes on the server and used as distillation targets (weight 0.5); heads stay local.
- **ZeroDFL:** fixed ring topology (two neighbours) defined before training; gossip averaging of the shared parameters
  only; no server. The prompt vector is added with unit scaling (λ = 1); for T5 the prompt is given in the input text.
- **TACTFL:** stage 1 InfoNCE (τ = 0.2, in-batch negatives) with similarity-aware server aggregation; stage 2 frozen
  encoder and a local head trained on a fixed 20 % labelled subset of each client. No balancing weight α.
- **LoRA:** r = 8, scaling 16, dropout 0.05 on `query`/`value` (`q`/`v` for T5).
- **Hyperparameter selection:** prior literature / backbone defaults (LoRA r = 8, AdamW 2e-5); no grid search or
  tuning on the test splits.
