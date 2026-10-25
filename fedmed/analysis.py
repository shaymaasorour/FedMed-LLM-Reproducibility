"""Part 3 analyses: text statistics, repeated-run statistics, significance tests, calibration, class imbalance,
non-LLM baselines and the edge-hardware cost of local LoRA fine-tuning."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import recall_score
from sklearn.preprocessing import OrdinalEncoder
from .config import CFG
from . import llm
from .llm import LABEL_COL, TEXT_COL
from .tabular import (DEVICE, SHARED_DIM, ClientModel, average_states, client_loader, full_metrics, load_shared,
                      pooled_metrics, predict_proba, shared_state, train_local)


# ---------------------------------------------------------------- 3.1 free-text statistics
def text_statistics(tokenizer, max_len=None):
    max_len = max_len or CFG["stage_B_llm"]["max_len"]
    rows = []
    for cid, c in llm.llm_clients.items():
        df = pd.concat([c["train"], c["test"]])
        text = df[TEXT_COL].fillna("").astype(str).str.strip()
        has_text = text.str.len() > 0
        n_tok = np.array([len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in text[has_text]])
        rows.append({"Client": cid, "Records": len(df), "With free text (%)": round(100 * has_text.mean(), 1),
                     "Mean tokens": round(n_tok.mean(), 1), "Median tokens": int(np.median(n_tok)),
                     "P95 tokens": int(np.percentile(n_tok, 95)), "Max tokens": int(n_tok.max()),
                     f"Truncated @{max_len} (%)": round(100 * (n_tok > max_len).mean(), 2),
                     "Structured fields": len(c["struct_cols"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3.2 mean +- SD and 95 % CI
def mean_sd(x):
    return f"{np.mean(x):.4f} ± {np.std(x, ddof=1):.4f}"


def ci95(x):
    half = stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))
    return f"[{np.mean(x) - half:.4f}, {np.mean(x) + half:.4f}]"


def summarize_runs(runs):
    """runs: one row per (Backbone, Method, Seed) with the federated Accuracy, F1 and ROC-AUC."""
    return (runs.groupby(["Backbone", "Method"], sort=False)
                .agg(**{"Accuracy (mean ± SD)": ("Accuracy", mean_sd), "Accuracy 95% CI": ("Accuracy", ci95),
                        "F1 (mean ± SD)": ("F1", mean_sd), "F1 95% CI": ("F1", ci95),
                        "ROC-AUC (mean ± SD)": ("ROC-AUC", mean_sd)})
                .reset_index())


# ---------------------------------------------------------------- 3.3 paired significance tests
def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def significance(runs, reference="TACTFL", others=("FedAvg", "FedProx", "Fed-GDBD", "ZeroDFL", "Centralized")):
    """Paired t-test and exact Wilcoxon over seeds (same seed = same initialisation), Holm-corrected per backbone."""
    rows = []
    for backbone in runs.Backbone.unique():
        ref = runs[(runs.Backbone == backbone) & (runs.Method == reference)].sort_values("Seed")["Accuracy"].values
        block = []
        for other in others:
            x = runs[(runs.Backbone == backbone) & (runs.Method == other)].sort_values("Seed")["Accuracy"].values
            t, p_t = stats.ttest_rel(ref, x)
            block.append({"Backbone": backbone, "Comparison": f"{reference} vs {other}",
                          "Mean Δ Acc": round(float(np.mean(ref - x)), 4), "t": round(float(t), 2),
                          "p (paired t)": p_t, "p (Wilcoxon)": stats.wilcoxon(ref, x).pvalue})
        for row, p_adj in zip(block, holm([b["p (paired t)"] for b in block])):
            row["p (Holm)"] = p_adj
            row["Significant (α=0.05)"] = "yes" if p_adj < 0.05 else "no"
        rows += block
    return pd.DataFrame(rows)


def paired_difference(fed, cen):
    """Federated vs centralized over independent runs (Table 9): mean difference, 95 % CI, paired t and Wilcoxon."""
    fed, cen = np.asarray(fed), np.asarray(cen)
    diff = fed - cen
    half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
    t_stat, p_t = stats.ttest_rel(fed, cen)
    return {"diff": diff.mean(), "ci": (diff.mean() - half, diff.mean() + half), "t": t_stat, "p_t": p_t,
            "p_wilcoxon": stats.wilcoxon(fed, cen).pvalue}


# ---------------------------------------------------------------- 3.4 calibration and class imbalance
def apply_temperature(logp, T):
    q = np.exp((logp - logp.max(1, keepdims=True)) / T)
    return q / q.sum(1, keepdims=True)


def fit_temperature(y, logp):
    nll = lambda T: -np.mean(np.log(apply_temperature(logp, T)[np.arange(len(y)), y] + 1e-12))
    return minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x


def temperature_scale(y, prob, seed=0):
    """Two-fold cross-fitted temperature scaling: T is fitted on one half of a client's test predictions and applied
    to the other half (and vice versa), so the calibrated ECE is never measured on the data used to fit T."""
    logp = np.log(np.clip(prob, 1e-12, 1.0))
    halves = np.array_split(np.random.RandomState(seed).permutation(len(y)), 2)
    out, Ts = np.empty_like(prob), []
    for fit, app in [(halves[0], halves[1]), (halves[1], halves[0])]:
        Ts.append(fit_temperature(y[fit], logp[fit]))
        out[app] = apply_temperature(logp[app], Ts[-1])
    return float(np.mean(Ts)), out


def min_class_recall(preds):
    return np.mean([min(recall_score(y, p.argmax(1), average=None, zero_division=0)) for y, p in preds.values()])


def calibration_row(preds_per_seed):
    """preds_per_seed: list (one per seed) of {client_id: (y, prob)} from the federated evaluation."""
    per_seed = []
    for preds in preds_per_seed:
        ms, ms_cal, Ts = [], [], []
        for y, prob in preds.values():
            T, prob_cal = temperature_scale(y, prob)
            ms.append(full_metrics(y, prob)); ms_cal.append(full_metrics(y, prob_cal)); Ts.append(T)
        per_seed.append({"ECE": np.mean([m["ECE"] for m in ms]), "ECE (temp-scaled)": np.mean([m["ECE"] for m in ms_cal]),
                         "Brier": np.mean([m["Brier"] for m in ms]), "Temperature": np.mean(Ts),
                         "Balanced Acc": np.mean([m["Balanced Acc"] for m in ms]),
                         "Macro-F1": np.mean([m["Macro-F1"] for m in ms]), "Min-class recall": min_class_recall(preds)})
    return pd.DataFrame(per_seed).mean().round(4).to_dict()


def downsample_minority(df, keep_ratio, seed):
    counts = df[LABEL_COL].value_counts()
    drop = df[df[LABEL_COL] == counts.idxmin()].sample(frac=1 - keep_ratio, random_state=seed).index
    return df.drop(index=drop).reset_index(drop=True)


def imbalance_stress_test(backbone, seeds, keep_ratios=(1.0, 0.5, 0.25, 0.1), method="tactfl"):
    """Down-sample the minority class of every client's training split and re-run the federation."""
    original = {cid: c["train"].copy() for cid, c in llm.llm_clients.items()}
    rows = []
    try:
        for keep in keep_ratios:
            per_seed = []
            for seed in seeds:
                for cid in llm.llm_clients:
                    llm.llm_clients[cid]["train"] = (downsample_minority(original[cid], keep, seed) if keep < 1
                                                     else original[cid])
                _, preds = llm.run_federated(backbone, method=method, seed=seed, return_predictions=True)
                ms = [full_metrics(y, p) for y, p in preds.values()]
                per_seed.append({"Balanced Acc": np.mean([m["Balanced Acc"] for m in ms]),
                                 "Macro-F1": np.mean([m["Macro-F1"] for m in ms]),
                                 "Min-class recall": min_class_recall(preds), "ECE": np.mean([m["ECE"] for m in ms])})
            ratio = np.mean([c["train"][LABEL_COL].value_counts().max() / c["train"][LABEL_COL].value_counts().min()
                             for c in llm.llm_clients.values()])
            rows.append({"Minority keep ratio": keep, "Mean imbalance ratio": round(ratio, 1),
                         **pd.DataFrame(per_seed).mean().round(4).to_dict()})
    finally:
        for cid in llm.llm_clients:
            llm.llm_clients[cid]["train"] = original[cid]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3.5 non-LLM baselines on the same clients
def structured_matrix(c):
    cols = c["struct_cols"]
    tr, te = c["train"][cols].copy(), c["test"][cols].copy()
    cat = [k for k in cols if tr[k].dtype == object]
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr[cat] = enc.fit_transform(tr[cat].astype(str)); te[cat] = enc.transform(te[cat].astype(str))
    med = tr.median()
    tr, te = tr.fillna(med).values.astype(np.float32), te.fillna(med).values.astype(np.float32)
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    return (tr - mu) / sd, (te - mu) / sd


def structured_specs():
    specs = {}
    for cid, c in llm.llm_clients.items():
        Xtr, Xte = structured_matrix(c)
        specs[cid] = dict(X_train=Xtr, X_test=Xte, E_train=np.zeros((len(Xtr), 0), np.int64),
                          E_test=np.zeros((len(Xte), 0), np.int64), y_train=c["train"][LABEL_COL].values,
                          y_test=c["test"][LABEL_COL].values, n_classes=c["n_classes"], cardinalities=[])
    return specs


def fed_mlp(specs, seed, cfg=CFG):
    """FedAvg over a shared MLP encoder, local input adapters + local heads (structured fields)."""
    torch.manual_seed(seed); np.random.seed(seed)
    A = cfg["stage_A_tabular"]
    models = {cid: ClientModel(s).to(DEVICE) for cid, s in specs.items()}
    g = shared_state(next(iter(models.values())))
    for _ in range(A["rounds"]):
        ups = []
        for cid, s in specs.items():
            load_shared(models[cid], g)
            train_local(models[cid], client_loader(s, bs=A["batch_size"]), A["local_epochs"], A["client_lr"])
            ups.append(shared_state(models[cid]))
        g = average_states(ups, [len(s["y_train"]) for s in specs.values()])
    res = []
    for cid, s in specs.items():
        load_shared(models[cid], g)
        res.append(full_metrics(*predict_proba(models[cid], s)))
    return pooled_metrics(res, [len(s["y_test"]) for s in specs.values()])


class GRUClient(nn.Module):
    def __init__(self, d, n_classes, hid=64):
        super().__init__()
        self.inp = nn.Linear(d, SHARED_DIM)                       # LOCAL
        self.encoder = nn.GRU(SHARED_DIM, hid, batch_first=True)   # SHARED
        self.head = nn.Linear(hid, n_classes)                      # LOCAL

    def forward(self, x):
        _, h = self.encoder(F.relu(self.inp(x)))
        return self.head(h[-1])


def seq_windows(X, y, w=8):
    idx = np.arange(len(y) - w + 1)[:, None] + np.arange(w)[None, :]
    return torch.tensor(X[idx]), torch.tensor(y[idx[:, -1]], dtype=torch.long)


def fed_gru(specs, seed, cfg=CFG):
    """FedAvg over a shared GRU encoder on windows of consecutive structured records."""
    torch.manual_seed(seed); np.random.seed(seed)
    A = cfg["stage_A_tabular"]
    data = {cid: (seq_windows(s["X_train"], s["y_train"]), seq_windows(s["X_test"], s["y_test"])) for cid, s in specs.items()}
    models = {cid: GRUClient(s["X_train"].shape[1], s["n_classes"]).to(DEVICE) for cid, s in specs.items()}
    g = shared_state(next(iter(models.values())))
    for _ in range(A["rounds"]):
        ups = []
        for cid, ((Xw, yw), _) in data.items():
            m = models[cid]; load_shared(m, g); m.train()
            opt = torch.optim.Adam(m.parameters(), lr=A["client_lr"])
            for _ in range(A["local_epochs"]):
                for xb, yb in DataLoader(TensorDataset(Xw, yw), batch_size=A["batch_size"], shuffle=True):
                    loss = F.cross_entropy(m(xb.to(DEVICE)), yb.to(DEVICE))
                    opt.zero_grad(); loss.backward(); opt.step()
            ups.append(shared_state(m))
        g = average_states(ups, [len(d[0][1]) for d in data.values()])
    res, sizes = [], []
    for cid, (_, (Xt, yt)) in data.items():
        m = models[cid]; load_shared(m, g); m.eval()
        with torch.no_grad():
            prob = F.softmax(m(Xt.to(DEVICE)), 1).cpu().numpy()
        res.append(full_metrics(yt.numpy(), prob)); sizes.append(len(yt))
    return pooled_metrics(res, sizes)


def local_xgboost(specs, seed):
    """XGBoost trained separately inside each hospital (local-only reference, no federation)."""
    from xgboost import XGBClassifier
    res, sizes = [], []
    for s in specs.values():
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                            colsample_bytree=0.8, random_state=seed, eval_metric="mlogloss")
        clf.fit(s["X_train"], s["y_train"])
        res.append(full_metrics(s["y_test"], clf.predict_proba(s["X_test"]))); sizes.append(len(s["y_test"]))
    return pooled_metrics(res, sizes)


# ---------------------------------------------------------------- 3.7 edge-hardware cost of local LoRA fine-tuning
def edge_cost(geometry, cfg=CFG):
    """geometry: {backbone name: backbone_geometry(...)} -> trainable parameters, upload size and peak memory."""
    Bc = cfg["stage_B_llm"]
    GB = 1024 ** 3
    lora_params = lambda g: g["layers"] * 2 * (g["hidden"] * Bc["lora_r"] + Bc["lora_r"] * g["hidden"])  # q and v

    def activation_bytes(g, batch, seq, checkpointing=False):
        s, b, h, a, L = seq, batch, g["hidden"], g["heads"], g["layers"]
        per_layer = s * b * h * (34 + 5 * a * s / h)                     # fp16 transformer activations
        return (L * 2 * s * b * h + per_layer) if checkpointing else L * per_layer

    rows = []
    for name, g in geometry.items():
        n_lora = lora_params(g)
        frozen = g["params"] * 2                                         # fp16 frozen backbone
        trainable = n_lora * 16                                          # fp32 weights + grads + Adam m, v
        rows.append({
            "Backbone": name,
            "LoRA trainable params": f"{n_lora:,}",
            "Trainable share (%)": round(100 * n_lora / g["params"], 3),
            "Upload per round (MB, fp32)": round(n_lora * 4 / 1024 ** 2, 2),
            "Frozen weights (GB, fp16)": round(frozen / GB, 2),
            f"Peak memory bs={Bc['batch_size']} (GB)":
                round((frozen + trainable + activation_bytes(g, Bc["batch_size"], Bc["max_len"])) / GB, 2),
            "Peak memory bs=4 + checkpointing (GB)":
                round((frozen + trainable + activation_bytes(g, 4, Bc["max_len"], checkpointing=True)) / GB, 2),
        })
    return pd.DataFrame(rows)
