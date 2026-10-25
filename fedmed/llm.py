"""Part 2: federated LLM engine (LoRA adapters shared, task heads local)."""
import os, copy, math, random
from .config import CFG
from .tabular import full_metrics, pooled_metrics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

B = CFG["stage_B_llm"]

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# 1 dataset = 1 hospital client. The five CSV files are NEVER concatenated;
# every client keeps its own label space, train/test split and task head.
# =====================================================================
# dataset_k.csv is the text export (free-text notes + structured fields) of a sample of records from the k-th hospital
# used in Part 1: 1 Hospital_Emergency, 2 Triage_History, 3 IERAD, 4 Patient_Data, 5 Healthcare.
CLIENT_FILES = {f"Client_{k}": f"dataset_{k}.csv" for k in range(1, 6)}
TEXT_COL, LABEL_COL = "text", "label"

def load_client(path, seed=42):
    df = pd.read_csv(path).dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    label_enc = LabelEncoder()
    df[LABEL_COL] = label_enc.fit_transform(df[LABEL_COL].astype(str))           # local label space
    struct_cols = [c for c in df.columns if c not in (TEXT_COL, LABEL_COL)]       # structured fields
    tr, te = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[LABEL_COL])
    return {"train": tr.reset_index(drop=True), "test": te.reset_index(drop=True),
            "n_classes": int(df[LABEL_COL].nunique()), "struct_cols": struct_cols,
            "classes": list(label_enc.classes_)}

llm_clients, LLM_CLIENTS = {}, []

def init_clients(data_dir=".", seed=42):
    """Load the five client CSV files (each one separately) into the module state."""
    global llm_clients, LLM_CLIENTS
    llm_clients = {cid: load_client(os.path.join(data_dir, path), seed) for cid, path in CLIENT_FILES.items()}
    LLM_CLIENTS = list(llm_clients)
    return llm_clients

def serialize_structured(row, cols):
    return "; ".join(f"{c.replace('_', ' ')}: {row[c]}" for c in cols if pd.notna(row[c]))

def make_inputs(df, struct_cols, mode="text", prefix=""):
    """mode = 'text' (free text only) | 'structured' (serialised fields only) | 'both'."""
    if mode == "text":
        texts = df[TEXT_COL].astype(str)
    elif mode == "structured":
        texts = df.apply(lambda r: serialize_structured(r, struct_cols), axis=1)
    else:
        texts = df.apply(lambda r: serialize_structured(r, struct_cols) + " | " + str(r[TEXT_COL]), axis=1)
    return [prefix + t for t in texts]

class TextDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts, self.labels, self.tok, self.max_len = texts, list(labels), tok, max_len
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len)
        enc["labels"] = int(self.labels[i])
        return enc

# =====================================================================
# Model: LoRA-adapted backbone (SHARED adapters) + local task heads
# =====================================================================
MODEL_REGISTRY = {
    "T5-Large": "google-t5/t5-large",
    "GLM-RoBERTa": "zai-org/glm-roberta-large",
    "RoBERTa-Large": "FacebookAI/roberta-large",
}
METHOD_NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedgdbd": "Fed-GDBD",
                "zerodfl": "ZeroDFL", "tactfl": "TACTFL", "centralized": "Centralized", "extractor": "Extractor-Only"}
RESULT_COLS = ["Accuracy", "Recall", "Precision", "F1", "ROC-AUC", "Loss", "R2", "RMSE"]
PLOT_METRICS = ["Accuracy", "Recall", "Precision", "F1", "ROC-AUC", "R2"]
SERVER_METHODS = {"fedavg", "fedprox", "fedgdbd", "tactfl"}
T5_PROMPT = "classify emergency severity: "      # ZeroDFL with T5: prompting through the textual input

def load_backbone(model_key):
    name = MODEL_REGISTRY[model_key]
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    is_t5 = "t5" in name.lower()
    base = (T5EncoderModel.from_pretrained(name) if is_t5
            else AutoModel.from_pretrained(name, trust_remote_code=True))
    lora = LoraConfig(r=B["lora_r"], lora_alpha=B["lora_scaling"], lora_dropout=B["lora_dropout"],
                      target_modules=["q", "v"] if is_t5 else ["query", "value"])
    return tok, get_peft_model(base, lora).to(device), base.config.hidden_size, is_t5

def pool(out, mask):
    h = out.last_hidden_state
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

def shared_of(encoder):
    return {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items() if "lora_" in k}

def load_shared_llm(encoder, state):
    encoder.load_state_dict(state, strict=False)

def weighted_average(states, weights=None):
    w = np.ones(len(states)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    return {k: sum(wi * s[k].float() for wi, s in zip(w, states)) for k in states[0]}

def ring_neighbours(ids):
    return {c: [ids[(k - 1) % len(ids)], ids[(k + 1) % len(ids)]] for k, c in enumerate(ids)}

def info_nce_llm(z1, z2, tau):
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2])
    sim = z @ z.t() / tau
    sim.masked_fill_(torch.eye(len(z), device=z.device, dtype=torch.bool), -9e15)
    n = z1.size(0)
    targets = torch.cat([torch.arange(n) + n, torch.arange(n)]).to(z.device)
    return F.cross_entropy(sim, targets)

# =====================================================================
# Controlled heterogeneity INSIDE each client (clients stay = datasets; no re-partitioning)
# =====================================================================
def skew_labels(df, alpha, seed=42):
    """Resample one client's training split to a Dirichlet(alpha) class mix.
    alpha=None -> natural distribution; large alpha -> close to balanced; small alpha -> strongly skewed."""
    if alpha is None:
        return df
    rng = np.random.RandomState(seed)
    counts = df[LABEL_COL].value_counts().sort_index()
    take = np.maximum(1, np.round(rng.dirichlet(alpha * np.ones(len(counts))) * len(df)).astype(int))
    parts = [df[df[LABEL_COL] == c].sample(n=t, replace=t > counts[c], random_state=seed)
             for c, t in zip(counts.index, take)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def label_emd(orig_labels, new_labels, n_classes):
    """1-D earth mover's distance between two label distributions of the same client."""
    from scipy.stats import wasserstein_distance
    support = np.arange(n_classes)
    return wasserstein_distance(support, support, np.bincount(orig_labels, minlength=n_classes) + 1e-12,
                                np.bincount(new_labels, minlength=n_classes) + 1e-12)

def state_distance(a, b):
    """Relative L2 distance ||a - b|| / ||b|| between two parameter dictionaries."""
    num = sum(((a[k].float() - b[k].float()) ** 2).sum() for k in b)
    den = sum((b[k].float() ** 2).sum() for k in b)
    return float(torch.sqrt(num / den.clamp(min=1e-12)))

# =====================================================================
# Federated engine
#   fedavg / fedprox / fedgdbd / tactfl : server-coordinated (shared LoRA adapters only)
#   zerodfl                             : peer-to-peer ring gossip, NO server
#   centralized                         : pooled multi-task reference (not federated)
#   extractor                           : frozen backbone, local heads only (no federation)
# =====================================================================
def run_federated(model_key, method="fedavg", label=None, rounds=None, input_mode="text",
                  seed=42, train_fraction=1.0, return_predictions=False, overrides=None,
                  label_skew_alpha=None, track_history=False, return_states=False):
    """overrides: dict of stage-B hyperparameters to change for this run (e.g. {"tactfl_tau": 0.5}).
    label_skew_alpha: controlled within-client label skew (see skew_labels).
    track_history: per-round federated / client accuracy and client drift.  return_states: final adapters + heads."""
    seed_everything(seed)
    Bx = {**B, **(overrides or {})}
    rounds = rounds or Bx["rounds"]
    label = label or f"{model_key} + {METHOD_NAMES[method]}"
    tok, encoder, hidden, is_t5 = load_backbone(model_key)
    prefix = T5_PROMPT if (is_t5 and method == "zerodfl") else ""
    collate = DataCollatorWithPadding(tok)

    loaders, test_loaders, lab_loaders, n_train = {}, {}, {}, {}
    for cid, c in llm_clients.items():
        tr = c["train"].sample(frac=train_fraction, random_state=seed) if train_fraction < 1 else c["train"]
        tr = skew_labels(tr, label_skew_alpha, seed)
        n_train[cid] = len(tr)
        loaders[cid] = DataLoader(TextDataset(make_inputs(tr, c["struct_cols"], input_mode, prefix), tr[LABEL_COL], tok, Bx["max_len"]),
                                  batch_size=Bx["batch_size"], shuffle=True, collate_fn=collate)
        test_loaders[cid] = DataLoader(TextDataset(make_inputs(c["test"], c["struct_cols"], input_mode, prefix), c["test"][LABEL_COL], tok, Bx["max_len"]),
                                       batch_size=2 * Bx["batch_size"], shuffle=False, collate_fn=collate)
        lab = tr.sample(frac=Bx["tactfl_label_fraction"], random_state=seed)        # fixed labelled subset (TACTFL stage 2)
        lab_loaders[cid] = DataLoader(TextDataset(make_inputs(lab, c["struct_cols"], input_mode, prefix), lab[LABEL_COL], tok, Bx["max_len"]),
                                      batch_size=2 * Bx["batch_size"], shuffle=False, collate_fn=collate)

    heads = {cid: nn.Linear(hidden, c["n_classes"]).to(device) for cid, c in llm_clients.items()}      # LOCAL
    prompts = {cid: nn.Parameter(torch.zeros(hidden, device=device)) for cid in llm_clients}          # LOCAL (ZeroDFL)
    use_prompt = (method == "zerodfl") and not is_t5

    def represent(batch, cid):
        z = pool(encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]), batch["attention_mask"])
        return z + Bx["zerodfl_prompt_scale_lambda"] * prompts[cid] if use_prompt else z

    def local_epoch(cid, global_state=None, proto_targets=None, contrastive=False):
        lora_params = [] if method == "extractor" else [p for n, p in encoder.named_parameters() if "lora_" in n]
        params = lora_params + ([] if contrastive else list(heads[cid].parameters()) + ([prompts[cid]] if use_prompt else []))
        opt = torch.optim.AdamW(params, lr=Bx["lr"], weight_decay=Bx["weight_decay"])
        encoder.train()
        for _ in range(Bx["local_epochs"]):
            for batch in loaders[cid]:
                batch = {k: v.to(device) for k, v in batch.items()}
                if contrastive:                                   # TACTFL stage 1 (two dropout views)
                    loss = info_nce_llm(represent(batch, cid), represent(batch, cid), Bx["tactfl_tau"])
                else:
                    z = represent(batch, cid)
                    loss = F.cross_entropy(heads[cid](z), batch["labels"])
                    if method == "fedprox":
                        loss = loss + Bx["fedprox_mu"] / 2 * sum(((p - global_state[n].to(device)) ** 2).sum()
                                                                for n, p in encoder.named_parameters() if "lora_" in n)
                    if method == "fedgdbd" and proto_targets is not None:
                        t = proto_targets[cid]
                        loss = loss + Bx["gdbd_proto_weight"] * F.mse_loss(z, t[batch["labels"]])
                opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def embed_all(cid, loader):
        encoder.eval()
        Z, Y = [], []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            Z.append(represent(batch, cid).cpu()); Y.append(batch["labels"].cpu())
        return torch.cat(Z), torch.cat(Y)

    def prototypes(cid):
        Z, Y = embed_all(cid, loaders[cid])
        return {int(c): Z[Y == c].mean(0) for c in torch.unique(Y)}

    def boundary_targets(bank, temp=0.5):
        keys = [(cid, c) for cid in bank for c in bank[cid]]
        P = torch.stack([bank[cid][c] for cid, c in keys]); Pn = F.normalize(P, dim=1)
        G = torch.softmax(Pn @ Pn.t() / temp, dim=1) @ P
        out = {cid: torch.zeros(llm_clients[cid]["n_classes"], hidden) for cid in bank}
        for (cid, c), g in zip(keys, G):
            out[cid][c] = g
        return {cid: t.to(device) for cid, t in out.items()}

    def fit_head_frozen(cid):                                      # TACTFL stage 2: frozen encoder, fixed labelled subset
        Z, Y = embed_all(cid, lab_loaders[cid])
        Z, Y = Z.to(device), Y.to(device)
        heads[cid].reset_parameters()                              # the head only ever sees the labelled subset
        opt = torch.optim.AdamW(heads[cid].parameters(), lr=1e-3)
        for _ in range(Bx["tactfl_head_epochs"]):
            loss = F.cross_entropy(heads[cid](Z), Y)
            opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def evaluate(cid):
        Z, Y = embed_all(cid, test_loaders[cid])
        prob = F.softmax(heads[cid](Z.to(device)), dim=1).cpu().numpy()
        return Y.numpy(), prob

    # ------------------------------ training ------------------------------
    global_state = shared_of(encoder)
    local_state = {cid: copy.deepcopy(global_state) for cid in LLM_CLIENTS}
    last_local = dict(local_state)                                  # adapters right after local training
    neighbours = ring_neighbours(LLM_CLIENTS)                       # fixed before training (ZeroDFL)
    targets = None
    history = []

    def round_snapshot(r):
        """Per-round federated accuracy, mean client accuracy and client drift."""
        fed, cli, sizes, drift = [], [], [], []
        for cid in LLM_CLIENTS:
            agg = local_state[cid] if method == "zerodfl" else global_state
            drift.append(state_distance(last_local[cid], agg))
            for state, bucket in ((last_local[cid], cli), (agg, fed)):
                load_shared_llm(encoder, state)
                if method == "tactfl":
                    fit_head_frozen(cid)
                y, prob = evaluate(cid)
                bucket.append((prob.argmax(1) == y).mean())
            sizes.append(len(y))
        history.append({"Round": r + 1, "Federated Acc": np.average(fed, weights=sizes),
                        "Client Acc": np.average(cli, weights=sizes), "Drift": float(np.mean(drift))})

    for r in range(rounds):
        if method in ("centralized", "extractor"):                 # no coordination between clients
            for cid in LLM_CLIENTS:
                local_epoch(cid)
            global_state = shared_of(encoder)
            local_state = {cid: global_state for cid in LLM_CLIENTS}
            last_local = dict(local_state)
            continue

        uploads, bank, summaries = [], {}, []
        for cid in LLM_CLIENTS:
            load_shared_llm(encoder, local_state[cid] if method == "zerodfl" else global_state)
            local_epoch(cid, global_state=global_state, proto_targets=targets, contrastive=(method == "tactfl"))
            local_state[cid] = shared_of(encoder)
            uploads.append(local_state[cid])
            if method == "fedgdbd":
                bank[cid] = prototypes(cid)
            if method == "tactfl":
                summaries.append(embed_all(cid, loaders[cid])[0].mean(0))

        last_local = dict(local_state)
        if method == "zerodfl":                                      # peer-to-peer gossip (no server)
            local_state = {cid: weighted_average([local_state[cid]] + [local_state[j] for j in neighbours[cid]])
                           for cid in LLM_CLIENTS}
        elif method == "tactfl":                                     # similarity-aware server aggregation
            S = F.normalize(torch.stack(summaries), dim=1)
            agreement = ((S @ S.t()).sum(1) - 1) / (len(summaries) - 1)
            global_state = weighted_average(uploads, torch.softmax(agreement / 0.1, 0).numpy())
        else:                                                        # FedAvg / FedProx / Fed-GDBD server
            global_state = weighted_average(uploads, [n_train[c] for c in LLM_CLIENTS])
            if method == "fedgdbd":
                targets = boundary_targets(bank)
        if track_history:
            round_snapshot(r)

    # ------------------------------ evaluation ------------------------------
    client_metrics, preds = [], {}
    for cid in LLM_CLIENTS:                                          # client-level: own local adapter state
        load_shared_llm(encoder, last_local[cid])
        if method == "tactfl":
            fit_head_frozen(cid)
        client_metrics.append(full_metrics(*evaluate(cid)))

    fed_per_client, sizes = [], []
    for cid in LLM_CLIENTS:                                          # federated: global / consensus adapters
        load_shared_llm(encoder, local_state[cid] if method == "zerodfl" else global_state)
        if method == "tactfl":
            fit_head_frozen(cid)
        y, prob = evaluate(cid)
        preds[cid] = (y, prob)
        fed_per_client.append(full_metrics(y, prob)); sizes.append(len(y))
    fed_metrics = pooled_metrics(fed_per_client, sizes)

    torch.cuda.empty_cache()                                         # encoder is freed when the function returns
    table = results_table(label, client_metrics, fed_metrics)
    if track_history or return_states:
        extras = {"predictions": preds, "history": pd.DataFrame(history),
                  "states": {"global": (weighted_average([local_state[c] for c in LLM_CLIENTS]) if method == "zerodfl"
                                        else global_state),
                             "local": last_local,
                             "heads": {cid: {k: v.cpu() for k, v in heads[cid].state_dict().items()} for cid in LLM_CLIENTS}}}
        return table, extras
    return (table, preds) if return_predictions else table

def results_table(label, client_metrics, fed_metrics):
    rows = [{"Model": label, "Client": f"Client_{i + 1}", **{c: m[c] for c in RESULT_COLS}}
            for i, m in enumerate(client_metrics)]
    rows.append({"Model": label, "Client": "Federated_Aggregate", **{c: fed_metrics[c] for c in RESULT_COLS}})
    return pd.DataFrame(rows).round(4)

def to_plot_df(results):
    out = results.set_index("Client")[PLOT_METRICS].copy()
    out.index = ["Federated" if c == "Federated_Aggregate" else c for c in out.index]
    return out

def federated_summary(results_by_model):
    return pd.DataFrame([{"Model": k, **r.iloc[-1][PLOT_METRICS].to_dict()} for k, r in results_by_model.items()])

def summary_dict(results_by_model):
    return {k: {"clients": r.iloc[:-1][PLOT_METRICS].values.astype(float),
                "federated": r.iloc[-1][PLOT_METRICS].values.astype(float)}
            for k, r in results_by_model.items()}

def print_results(df, title):
    print(f"\n=== {title} ===\n")
    print(df.to_string(index=False))

