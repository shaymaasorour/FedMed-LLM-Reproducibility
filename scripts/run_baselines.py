"""Section 3.5: non-LLM baselines on the structured fields of the same five clients
(Fed-MLP and Fed-GRU with FedAvg, local-only XGBoost), mean ± SD over the seeds."""
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedmed.config import CFG
from fedmed import analysis, llm

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=".", help="folder with dataset_1.csv ... dataset_5.csv")
p.add_argument("--out", default=None)
args = p.parse_args()

llm.init_clients(args.data_dir, seed=42)
specs = analysis.structured_specs()
rows = []
for name, kind, fn in [("Fed-MLP (FedAvg)", "non-LLM, federated", lambda s: analysis.fed_mlp(specs, s)),
                       ("Fed-GRU (FedAvg)", "non-LLM, federated, temporal", lambda s: analysis.fed_gru(specs, s)),
                       ("XGBoost (local only)", "non-LLM, no federation", lambda s: analysis.local_xgboost(specs, s))]:
    ms = [fn(seed) for seed in CFG["seeds"]]
    rows.append({"Model": name, "Type": kind,
                 **{f"{k} (mean ± SD)": analysis.mean_sd([m[k] for m in ms]) for k in ["Accuracy", "F1", "ROC-AUC"]}})
table = pd.DataFrame(rows)
print(table.to_string(index=False))
print("\nLLM rows (FedAvg / FedProx / TACTFL with the same clients): scripts/run_repeated.py")
if args.out:
    table.to_csv(args.out, index=False)
