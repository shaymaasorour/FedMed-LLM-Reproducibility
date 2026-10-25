"""Section 3.2: repeated independent runs (every seed x method x backbone), mean ± SD and 95 % CI.

Writes runs.csv (federated aggregate per run) and predictions.pkl (per-client test predictions, used by
run_statistics.py for the calibration analysis)."""
import argparse, os, pickle, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedmed.config import CFG
from fedmed import analysis, llm

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=".", help="folder with dataset_1.csv ... dataset_5.csv")
p.add_argument("--backbones", nargs="+", default=list(llm.MODEL_REGISTRY))
p.add_argument("--methods", nargs="+", default=["fedavg", "fedprox", "fedgdbd", "zerodfl", "tactfl", "centralized"])
p.add_argument("--seeds", nargs="+", type=int, default=CFG["seeds"])
p.add_argument("--out", default="results/repeated")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

llm.init_clients(args.data_dir, seed=42)
rows, predictions = [], {}
for backbone in args.backbones:
    for method in args.methods:
        for seed in args.seeds:
            table, preds = llm.run_federated(backbone, method=method, seed=seed, return_predictions=True)
            fed = table.iloc[-1]
            rows.append({"Backbone": backbone, "Method": llm.METHOD_NAMES[method], "Seed": seed,
                         **{m: fed[m] for m in ["Accuracy", "F1", "ROC-AUC"]}})
            predictions[(backbone, method, seed)] = preds
runs = pd.DataFrame(rows)
runs.to_csv(os.path.join(args.out, "runs.csv"), index=False)
with open(os.path.join(args.out, "predictions.pkl"), "wb") as f:
    pickle.dump(predictions, f)
print(analysis.summarize_runs(runs).to_string(index=False))
