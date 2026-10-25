"""Stage C (Figures 14-15): device population and round-by-round run of the federation for the three hybrids."""
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedmed.config import CFG
from fedmed.data_tabular import DATASET_FILES, build_clients
from fedmed import llm, systems, tabular

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="./datasets", help="folder with the five hospital CSV files (Part 1)")
p.add_argument("--text-dir", default=".", help="folder with dataset_1.csv ... dataset_5.csv (Part 2)")
p.add_argument("--backbone", choices=list(llm.MODEL_REGISTRY), default="T5-Large")
p.add_argument("--out", default="results/systems")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

client_data = tabular.check_clients(build_clients(args.data_dir))
registry = systems.build_device_registry({cid: len(client_data[cid]["y_train"]) for cid in tabular.CLIENTS})
print(f"Device population: {len(registry)} devices across {len(tabular.CLIENTS)} hospital clients\n")
print(pd.crosstab(registry["hospital"], registry["type"], margins=True, margins_name="Total").to_string())
registry.to_csv(os.path.join(args.out, "device_registry.csv"), index=False)

llm.init_clients(args.text_dir)
n_train = {cid: len(llm.llm_clients[cid]["train"]) for cid in tabular.CLIENTS}
emergency = pd.read_csv(os.path.join(args.data_dir, DATASET_FILES["Hospital_Emergency"]))
availability = 1.0 - systems.BUSY_AT_PEAK * systems.arrival_profile(emergency)
geo = systems.backbone_geometry(args.backbone)

for method in ["fedgdbd", "zerodfl", "tactfl"]:
    rounds, devices = systems.run_systems(method, registry, n_train, availability, geo, CFG)
    pd.DataFrame(rounds).rename_axis("hour").to_csv(os.path.join(args.out, f"rounds_per_hour_{method}.csv"))
    pd.DataFrame(devices).rename_axis("hour").to_csv(os.path.join(args.out, f"devices_per_round_{method}.csv"))
    print(f"{llm.METHOD_NAMES[method]:9s}: {int(rounds['Completed'].sum())} completed rounds in {CFG['stage_C_systems']['hours']} h")
