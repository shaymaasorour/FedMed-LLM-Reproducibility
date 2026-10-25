"""Stage C (Figures 14-15): device population of the five hospital clients and a round-by-round run of the federation.

Each hospital client operates a pool of devices (edge servers, clinical workstations, mobile units); a device only runs
the local update of its own hospital, so no new clients or data are introduced. Every round a device is available
unless it is busy with clinical work (hourly arrival profile of the Hospital Emergency dataset), may drop out and
retries once; compute time follows the training FLOPs of the backbone and communication time the LoRA adapter size.
Devices slower than the round deadline are cut; a round completes when every hospital delivers an update.
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, T5EncoderModel
from .config import CFG
from .llm import MODEL_REGISTRY

DEVICE_TYPES = {                 # share of devices, relative compute speed, uplink (Mbit/s), per-round drop-out probability
    "edge server":          {"share": 0.10, "speed": 4.0, "mbps": 100.0, "p_drop": 0.02},
    "clinical workstation": {"share": 0.55, "speed": 1.0, "mbps": 50.0, "p_drop": 0.05},
    "mobile unit":          {"share": 0.35, "speed": 0.4, "mbps": 10.0, "p_drop": 0.12},
}
MIN_PER_HOSPITAL = 20
REFERENCE_TFLOPS = 10.0      # sustained fp16 throughput of a clinical workstation (speed = 1.0)
UTILISATION = 0.3            # achieved fraction of that throughput during fine-tuning
DEADLINE_FACTOR = 2.0        # round deadline = 2 x median device time; slower devices are cut
BUSY_AT_PEAK = 0.5           # share of devices occupied by clinical work at the busiest ED hour


def build_device_registry(train_sizes, cfg=CFG, seed=2024):
    """train_sizes: {client_id: local training volume}. A minimum pool per hospital, the rest proportional to volume."""
    S = cfg["stage_C_systems"]
    rng = np.random.RandomState(seed)
    clients = list(train_sizes)
    sizes = np.array([train_sizes[c] for c in clients], dtype=float)
    extra = (S["devices"] - MIN_PER_HOSPITAL * len(clients)) * sizes / sizes.sum()
    alloc = MIN_PER_HOSPITAL + np.floor(extra).astype(int)
    for i in np.argsort(-(extra - np.floor(extra)))[: S["devices"] - alloc.sum()]:   # largest remainder
        alloc[i] += 1
    types, probs = list(DEVICE_TYPES), [v["share"] for v in DEVICE_TYPES.values()]
    return pd.DataFrame([
        {"device_id": f"{cid}-D{j:03d}", "hospital": cid, "type": t,
         "speed": DEVICE_TYPES[t]["speed"], "mbps": DEVICE_TYPES[t]["mbps"], "p_drop": DEVICE_TYPES[t]["p_drop"]}
        for cid, n in zip(clients, alloc)
        for j, t in enumerate(rng.choice(types, size=n, p=probs))
    ])


def backbone_geometry(model_key):
    """Parameter count and geometry of a backbone, read from its configuration (weights are not downloaded)."""
    name = MODEL_REGISTRY[model_key]
    cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
    with torch.device("meta"):
        net = T5EncoderModel(cfg) if "t5" in name.lower() else AutoModel.from_config(cfg, trust_remote_code=True)
    return {"params": sum(p.numel() for p in net.parameters()), "layers": cfg.num_hidden_layers,
            "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads}


def arrival_profile(df):
    """Hourly arrival share of the Hospital Emergency dataset, scaled to 1 at the busiest hour."""
    for c in df.columns:
        if any(k in c.lower() for k in ("arrival", "time", "date")):
            ts = pd.to_datetime(df[c], errors="coerce")
            if ts.notna().mean() > 0.8 and ts.dt.hour.nunique() > 1:
                p = ts.dt.hour.value_counts().reindex(range(24), fill_value=0).values.astype(float)
                return p / p.max()
    raise ValueError("no arrival timestamp column found in the Hospital Emergency dataset")


def method_cost(cfg=CFG):
    """Passes over the local data per round (a forward-only pass counts 1/3) and adapter transfers per round."""
    Bc = cfg["stage_B_llm"]
    return {
        "fedgdbd": {"passes": Bc["local_epochs"] + 1 / 3, "transfers": 2},       # local epoch + prototype pass; up + down
        "zerodfl": {"passes": Bc["local_epochs"], "transfers": 4},               # send to / receive from two ring neighbours
        "tactfl":  {"passes": 2 * Bc["local_epochs"] + 1 / 3, "transfers": 2},   # two contrastive views + summary pass
    }


def run_systems(method, registry, n_train, availability, geo, cfg=CFG, seed=0):
    """Round-by-round run over the configured hours.

    registry: device table (build_device_registry); n_train: {client_id: local training size of the LLM stage};
    availability: 24 hourly availability probabilities; geo: backbone_geometry(...).
    Returns (rounds per hour by outcome, mean device states per round in every hour)."""
    S, Bc = cfg["stage_C_systems"], cfg["stage_B_llm"]
    clients = list(n_train)
    reg = registry.reset_index(drop=True)
    hosp = reg["hospital"].map({cid: i for i, cid in enumerate(clients)}).values
    n_tr = np.array([n_train[cid] for cid in clients], dtype=float)
    speed, p_drop, mbps = reg["speed"].values, reg["p_drop"].values, reg["mbps"].values
    flops_per_sample = 6 * geo["params"] * Bc["max_len"]                                 # forward + backward
    adapter_mb = geo["layers"] * 2 * 2 * geo["hidden"] * Bc["lora_r"] * 4 / 1024 ** 2   # q/v LoRA adapters, fp32

    rng = np.random.RandomState(seed)
    cost, n_h = method_cost(cfg)[method], len(clients)
    rounds = {k: np.zeros(S["hours"]) for k in ["Completed", "Failed", "Retried", "Aborted"]}
    devices = {k: [[] for _ in range(S["hours"])] for k in ["Dropped Out", "Completed", "Aborted", "Busy"]}
    t = 0.0
    while t < S["hours"] * 3600:
        hour = int(t // 3600)
        avail = rng.rand(len(reg)) < availability[hour % 24]
        n_avail = np.bincount(hosp[avail], minlength=n_h)
        samples = n_tr[hosp] / np.maximum(n_avail[hosp], 1)                    # local data split over available devices
        base = (samples * cost["passes"] * flops_per_sample / (speed * REFERENCE_TFLOPS * 1e12 * UTILISATION)
                + cost["transfers"] * adapter_mb * 8 / mbps)
        dropped = avail & (rng.rand(len(reg)) < p_drop)
        retried = dropped & (rng.rand(len(reg)) >= p_drop)                     # one retry per dropped device
        alive = avail & (~dropped | retried)
        finish = np.where(retried, 2 * base, base)
        deadline = DEADLINE_FACTOR * np.median(base[avail]) if avail.any() else 60.0
        done = alive & (finish <= deadline)
        cut = alive & ~done

        if (np.bincount(hosp[done], minlength=n_h) > 0).all():
            rounds["Completed"][hour] += 1
            rounds["Retried"][hour] += int((np.bincount(hosp[done & ~retried], minlength=n_h) == 0).any())
        elif (np.bincount(hosp[alive], minlength=n_h) == 0).any():
            rounds["Failed"][hour] += 1            # a hospital lost all its devices: no update this round
        else:
            rounds["Aborted"][hour] += 1           # a hospital's devices all missed the deadline
        for k, mask in [("Dropped Out", dropped & ~retried), ("Completed", done), ("Aborted", cut), ("Busy", ~avail)]:
            devices[k][hour].append(mask.sum())
        t += deadline if cut.any() or not done.any() else max(finish[done].max(), 1.0)
    return rounds, {k: np.array([np.mean(v) if v else 0.0 for v in vs]) for k, vs in devices.items()}
