"""Sections 3.3-3.4: paired significance tests (paired t, exact Wilcoxon, Holm), calibration
(ECE, cross-fitted temperature scaling, Brier, per-class recall) and, with --imbalance, the class-imbalance stress test."""
import argparse, os, pickle, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedmed import analysis, llm

p = argparse.ArgumentParser()
p.add_argument("--runs", default="results/repeated", help="output folder of run_repeated.py")
p.add_argument("--backbone", default="GLM-RoBERTa", help="backbone for calibration and imbalance")
p.add_argument("--imbalance", action="store_true", help="also run the minority down-sampling stress test")
p.add_argument("--data-dir", default=".", help="folder with dataset_1.csv ... dataset_5.csv (imbalance test)")
args = p.parse_args()

runs = pd.read_csv(os.path.join(args.runs, "runs.csv"))
sig = analysis.significance(runs)
sig.to_csv(os.path.join(args.runs, "significance.csv"), index=False)
for col in ["p (paired t)", "p (Wilcoxon)", "p (Holm)"]:
    sig[col] = sig[col].map(lambda v: f"{v:.4f}")
print(f"=== Paired significance tests vs TACTFL (n = {runs.Seed.nunique()} seeds) ===\n")
print(sig.to_string(index=False))
print("\nWith n = 5 paired runs the smallest attainable two-sided exact Wilcoxon p-value is 0.0625.")

with open(os.path.join(args.runs, "predictions.pkl"), "rb") as f:
    predictions = pickle.load(f)
seeds = sorted(runs.Seed.unique())
cal = pd.DataFrame([{"Method": llm.METHOD_NAMES[m],
                     **analysis.calibration_row([predictions[(args.backbone, m, s)] for s in seeds])}
                    for m in ["fedavg", "fedprox", "fedgdbd", "zerodfl", "tactfl"]])
cal.to_csv(os.path.join(args.runs, "calibration.csv"), index=False)
print(f"\n=== Calibration and per-class behaviour ({args.backbone}, mean over {len(seeds)} seeds) ===\n")
print(cal.to_string(index=False))

if args.imbalance:
    llm.init_clients(args.data_dir, seed=42)
    imb = analysis.imbalance_stress_test(args.backbone, seeds)
    imb.to_csv(os.path.join(args.runs, "imbalance.csv"), index=False)
    print(f"\n=== Class-imbalance stress test (TACTFL + {args.backbone}) ===\n")
    print(imb.to_string(index=False))
