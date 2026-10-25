"""Write the per-client train/test splits used in the paper (tabular .npz + text .csv)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedmed.data_tabular import build_clients
from fedmed import llm

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="./datasets", help="folder with the five hospital CSV files (Part 1)")
p.add_argument("--text-dir", default=".", help="folder with dataset_1.csv ... dataset_5.csv (Part 2)")
p.add_argument("--out", default="./splits")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

for cid, c in build_clients(args.data_dir, seed=args.seed).items():
    np.savez_compressed(os.path.join(args.out, f"{cid}_tabular.npz"),
                        **{k: c[k] for k in ("X_train", "X_test", "E_train", "E_test", "y_train", "y_test")})
    print(f"{cid}: {c['source']} -> {len(c['y_train'])} train / {len(c['y_test'])} test")

for cid, c in llm.init_clients(args.text_dir, seed=args.seed).items():
    c["train"].to_csv(os.path.join(args.out, f"{cid}_text_train.csv"), index=False)
    c["test"].to_csv(os.path.join(args.out, f"{cid}_text_test.csv"), index=False)
