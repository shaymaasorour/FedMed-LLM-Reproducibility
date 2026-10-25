"""Run one Part 2 federated LLM experiment (one backbone, one strategy, one seed)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedmed import llm

p = argparse.ArgumentParser()
p.add_argument("--backbone", choices=list(llm.MODEL_REGISTRY), required=True)
p.add_argument("--method", choices=list(llm.METHOD_NAMES), required=True)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--data-dir", default=".", help="folder with dataset_1.csv ... dataset_5.csv")
p.add_argument("--input-mode", choices=["text", "structured", "both"], default="text")
p.add_argument("--train-fraction", type=float, default=1.0)
p.add_argument("--label-skew-alpha", type=float, default=None)
p.add_argument("--rounds", type=int, default=None)
p.add_argument("--tau", type=float, default=None, help="override TACTFL temperature")
p.add_argument("--label-fraction", type=float, default=None, help="override TACTFL labelled fraction")
p.add_argument("--out", default=None)
args = p.parse_args()

overrides = {k: v for k, v in {"tactfl_tau": args.tau, "tactfl_label_fraction": args.label_fraction}.items() if v is not None}
llm.init_clients(args.data_dir, seed=42)
table = llm.run_federated(args.backbone, args.method, seed=args.seed, input_mode=args.input_mode,
                          train_fraction=args.train_fraction, label_skew_alpha=args.label_skew_alpha,
                          rounds=args.rounds, overrides=overrides)
print(table.to_string(index=False))
if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out, index=False)
