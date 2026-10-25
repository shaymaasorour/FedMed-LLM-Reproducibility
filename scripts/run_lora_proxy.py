"""Stage D: LoRA hybrid tables (Table 1 Fed-GDBD / TreeXformer, Table 2 ZeroDFL / ClinicalBERT proxy,
Table 3 TACTFL / TFT proxy) and, with --table9, the federated-vs-centralized test over all seeds."""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedmed.config import CFG
from fedmed.data_tabular import build_clients
from fedmed import analysis, lora_proxy, tabular

TABLES = {1: ("tree", "server"), 2: ("bert", "gossip"), 3: ("tft", "server")}

p = argparse.ArgumentParser()
p.add_argument("--table", type=int, choices=list(TABLES), default=3)
p.add_argument("--data-dir", default="./datasets")
p.add_argument("--new-results", action="store_true", help="use CFG rounds_new_results instead of rounds")
p.add_argument("--table9", action="store_true", help="repeat Table 3 over every seed and test federated vs centralized")
p.add_argument("--out", default=None)
args = p.parse_args()

D = CFG["stage_D_lora_proxy"]
data = lora_proxy.prepare(tabular.check_clients(build_clients(args.data_dir)))
rounds = D["rounds_new_results"] if (args.new_results or args.table9) else D["rounds"]
backbone, aggregation = TABLES[args.table]

if not args.table9:
    table = lora_proxy.lora_results_table(lora_proxy.BACKBONES[backbone], data, CFG, rounds=rounds, aggregation=aggregation)
    print(table.to_string())
    if args.out:
        table.to_csv(args.out)
else:
    rows = []
    for seed in CFG["seeds"]:
        torch.manual_seed(seed); np.random.seed(seed)
        t = lora_proxy.lora_results_table(lora_proxy.TFTProxy, data, CFG, rounds=rounds)
        rows.append({"Seed": seed, "Federated": t.loc["Accuracy", "LoRA Federated"],
                     "Centralized": t.loc["Accuracy", "LoRA Centralized"]})
    t9 = pd.DataFrame(rows)
    r = analysis.paired_difference(t9["Federated"], t9["Centralized"])
    print(t9.round(4).to_string(index=False))
    print(f"\nFederated   : {analysis.mean_sd(t9['Federated'])}\nCentralized : {analysis.mean_sd(t9['Centralized'])}")
    print(f"Difference  : {r['diff']:+.4f}  (95% CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}])")
    print(f"Paired t-test: t = {r['t']:.2f}, p = {r['p_t']:.4f} | Wilcoxon signed-rank: p = {r['p_wilcoxon']:.4f}")
    if args.out:
        t9.to_csv(args.out, index=False)
