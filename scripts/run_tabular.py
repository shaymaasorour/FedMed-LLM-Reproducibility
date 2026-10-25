"""Run one Part 1 federated strategy on the five hospital clients."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedmed.config import CFG
from fedmed.data_tabular import build_clients
from fedmed import tabular

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["fedgdbd", "zerodfl", "tactfl"], required=True)
p.add_argument("--data-dir", default="./datasets")
p.add_argument("--out", default=None, help="CSV file for the per-round federated results")
args = p.parse_args()

clients = tabular.check_clients(build_clients(args.data_dir))
client_df, global_df = getattr(tabular, f"run_{args.method}")(clients, CFG)
print(client_df); print(global_df.to_string(index=False))
if args.out:
    global_df.to_csv(args.out, index=False)
