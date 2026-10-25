"""Part 1: local adapters + shared encoder + local heads, and the Fed-GDBD / ZeroDFL / TACTFL strategies."""
import random
import numpy as np
from .config import CFG
# ===== Federated building blocks =====
#   LocalInputAdapter : LOCAL  - maps each hospital's own schema (dense + embedded features) to a common space
#   SharedEncoder     : SHARED - the only parameters that take part in federated coordination
#   head              : LOCAL  - task-specific prediction head (never averaged across clients)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, r2_score)

np.random.seed(42)
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHARED_DIM = 64
HIDDEN_DIM = 128
CLIENTS = [f"Client_{k}" for k in range(1, CFG["num_clients"] + 1)]

# ---- every hospital client must come from its own dataset (5-client topology) ----
def check_clients(client_data):
    missing = [cid for cid in CLIENTS if cid not in client_data]
    if missing:
        raise RuntimeError(f"No data for {missing}: check the dataset files and target columns")
    return client_data

# ---- model components ----
class LocalInputAdapter(nn.Module):
    def __init__(self, dense_dim, cardinalities, out_dim=SHARED_DIM):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c + 1, min(50, (c + 1) // 2)) for c in cardinalities])
        emb_dim = sum(e.embedding_dim for e in self.embs)
        self.proj = nn.Linear(dense_dim + emb_dim, out_dim)

    def forward(self, x, e):
        parts = [x] + [emb(e[:, j]) for j, emb in enumerate(self.embs)]
        return F.relu(self.proj(torch.cat(parts, dim=1)))

class SharedEncoder(nn.Module):
    def __init__(self, in_dim=SHARED_DIM, hid=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.bn1 = nn.BatchNorm1d(hid)
        self.fc2 = nn.Linear(hid, hid)
        self.bn2 = nn.BatchNorm1d(hid)

    def forward(self, h):
        h = F.relu(self.bn1(self.fc1(h)))
        return F.relu(self.bn2(self.fc2(h)))

class ClientModel(nn.Module):
    def __init__(self, spec, use_prompt=False, prompt_scale=1.0):
        super().__init__()
        self.adapter = LocalInputAdapter(spec["X_train"].shape[1], spec["cardinalities"])
        self.encoder = SharedEncoder()
        self.head = nn.Linear(HIDDEN_DIM, spec["n_classes"])
        self.use_prompt, self.prompt_scale = use_prompt, prompt_scale
        self.prompt = nn.Parameter(torch.zeros(SHARED_DIM))       # LOCAL prompt vector (ZeroDFL)

    def forward(self, x, e, return_feat=False):
        h = self.adapter(x, e)
        if self.use_prompt:
            h = h + self.prompt_scale * self.prompt               # lambda = 1 (unit scaling)
        feat = self.encoder(h)
        logits = self.head(feat)
        return (logits, feat) if return_feat else logits

SHARED_PREFIX = "encoder."

def shared_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(SHARED_PREFIX)}

def load_shared(model, state):
    model.load_state_dict(state, strict=False)

def average_states(states, weights=None):
    weights = np.ones(len(states)) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return {k: sum(w * s[k].float() for w, s in zip(weights, states)) for k in states[0]}

def client_tensors(spec, split="train"):
    return (torch.tensor(spec[f"X_{split}"]), torch.tensor(spec[f"E_{split}"]),
            torch.tensor(spec[f"y_{split}"], dtype=torch.long))

def client_loader(spec, split="train", bs=128, shuffle=True):
    return DataLoader(TensorDataset(*client_tensors(spec, split)), batch_size=bs, shuffle=shuffle)

# ---- metrics (incl. calibration) ----
def ece_score(y_true, prob, n_bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y_true[m]).mean() - conf[m].mean())
    return ece

def full_metrics(y_true, prob):
    y_true = np.asarray(y_true)
    n_classes = prob.shape[1]
    y_pred = prob.argmax(1)
    expected = (prob * np.arange(n_classes)).sum(1)               # regression view on the class index
    try:
        auc = (roc_auc_score(y_true, prob[:, 1]) if n_classes == 2
               else roc_auc_score(y_true, prob, multi_class="ovr", labels=list(range(n_classes))))
    except ValueError:
        auc = np.nan
    onehot = np.eye(n_classes)[y_true]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "ROC-AUC": auc,
        "Loss": log_loss(y_true, prob, labels=list(range(n_classes))),
        "R2": r2_score(y_true, expected),
        "RMSE": np.sqrt(mean_squared_error(y_true, expected)),
        "MAE": mean_absolute_error(y_true, expected),
        "ECE": ece_score(y_true, prob),
        "Brier": np.mean(np.sum((prob - onehot) ** 2, axis=1)),
        "Balanced Acc": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

@torch.no_grad()
def predict_proba(model, spec, split="test", bs=1024):
    model.eval()
    X, E, y = client_tensors(spec, split)
    probs = [F.softmax(model(X[i:i + bs].to(DEVICE), E[i:i + bs].to(DEVICE)), dim=1).cpu().numpy()
             for i in range(0, len(y), bs)]
    return y.numpy(), np.vstack(probs)

def pooled_metrics(per_client, sizes):
    """Sample-weighted average of client metrics (no data leaves the clients)."""
    w = np.asarray(sizes, dtype=float) / np.sum(sizes)
    return {k: float(np.nansum([wi * m[k] for wi, m in zip(w, per_client)])) for k in per_client[0]}

def train_local(model, loader, epochs, lr, extra_loss=None, params=None):
    model.train()
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, eb, yb in loader:
            xb, eb, yb = xb.to(DEVICE), eb.to(DEVICE), yb.to(DEVICE)
            logits, feat = model(xb, eb, return_feat=True)
            loss = F.cross_entropy(logits, yb)
            if extra_loss is not None:
                loss = loss + extra_loss(feat, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

ROUND_COLS = ["Accuracy", "Recall", "Precision", "F1", "ROC-AUC", "Loss", "R2", "RMSE", "MAE"]

def print_round_table(df):
    print(f"{'Round':<7}| {'Accuracy':<10}| {'Recall':<10}| {'Precision':<11}| {'F1-Score':<10}| "
          f"{'ROC-AUC':<9}| {'Loss':<8}| {'R²':<7}| {'RMSE':<8}| {'MAE':<7}")
    print("-" * 105)
    for _, r in df.iterrows():
        print(f"{int(r['Round']):<7}| {r['Accuracy']:<10.3f}| {r['Recall']:<10.3f}| {r['Precision']:<11.3f}| "
              f"{r['F1-Score']:<10.3f}| {r['ROC-AUC']:<9.3f}| {r['Loss']:<8.3f}| {r['R²']:<7.3f}| "
              f"{r['RMSE']:<8.3f}| {r['MAE']:<7.3f}")

def to_round_df(rows):
    df = pd.DataFrame(rows)[["Round"] + ROUND_COLS].rename(columns={"F1": "F1-Score", "R2": "R²"})
    return df.round(3)

def client_mean_table(history):
    t = pd.DataFrame({cid: pd.DataFrame(h)[ROUND_COLS].mean() for cid, h in history.items()}).T.round(3)
    t.index.name = "Client"
    return t



def run_fedgdbd(client_data, CFG=CFG):
    # ============================================
    # Implementation — Fed-GDBD (server-coordinated)
    #   1. each hospital trains its local model (adapter + shared encoder + local head)
    #   2. each hospital uploads class-wise prototypes of its shared features + its shared-encoder update
    #   3. server: FedAvg of the shared encoder, prototypes -> global decision-boundary prototypes
    #   4. global prototypes are sent back and used as DISTILLATION TARGETS for the shared representation
    #      (prototypes are never converted into model weights; local heads are never averaged)
    # ============================================

    A = CFG["stage_A_tabular"]

    def set_seed(seed=42):
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    set_seed(42)

    @torch.no_grad()
    def class_prototypes(model, spec):
        model.eval()
        X, E, y = client_tensors(spec, "train")
        feats = torch.cat([model(X[i:i + 2048].to(DEVICE), E[i:i + 2048].to(DEVICE), return_feat=True)[1].cpu()
                           for i in range(0, len(y), 2048)])
        return {int(c): feats[y == c].mean(0) for c in torch.unique(y)}

    def global_boundary_targets(bank, temp=0.5):
        """Server: every (client, class) prototype attends over the whole prototype bank."""
        keys = [(cid, c) for cid, protos in bank.items() for c in protos]
        P = torch.stack([bank[cid][c] for cid, c in keys])
        Pn = F.normalize(P, dim=1)
        G = torch.softmax(Pn @ Pn.t() / temp, dim=1) @ P
        targets = {}
        for (cid, c), g in zip(keys, G):
            targets.setdefault(cid, {})[c] = g
        return targets

    models = {cid: ClientModel(client_data[cid]).to(DEVICE) for cid in CLIENTS}
    global_shared = shared_state(models[CLIENTS[0]])
    sizes_train = [len(client_data[cid]["y_train"]) for cid in CLIENTS]
    sizes_test = [len(client_data[cid]["y_test"]) for cid in CLIENTS]
    targets = None

    client_history = {cid: [] for cid in CLIENTS}
    global_rows = []

    for r in range(A["rounds"]):
        uploads, bank = [], {}
        for cid in CLIENTS:
            spec, model = client_data[cid], models[cid]
            load_shared(model, global_shared)                                   # receive global shared encoder

            tgt = None if targets is None else torch.stack(
                [targets[cid].get(c, torch.zeros(HIDDEN_DIM)) for c in range(spec["n_classes"])]).to(DEVICE)
            distill = None if tgt is None else (lambda feat, yb, tgt=tgt: A["gdbd_proto_weight"] * F.mse_loss(feat, tgt[yb]))

            train_local(model, client_loader(spec, bs=A["batch_size"]), A["local_epochs"], A["client_lr"], distill)
            client_history[cid].append(full_metrics(*predict_proba(model, spec)))   # client-level result

            uploads.append(shared_state(model))
            bank[cid] = class_prototypes(model, spec)

        # ----- server -----
        global_shared = average_states(uploads, weights=sizes_train)
        targets = global_boundary_targets(bank)

        # ----- federated evaluation: global shared encoder + each client's local head -----
        per_client = []
        for cid in CLIENTS:
            load_shared(models[cid], global_shared)
            per_client.append(full_metrics(*predict_proba(models[cid], client_data[cid])))
        global_rows.append({"Round": r + 1, **pooled_metrics(per_client, sizes_test)})

    client_df = client_mean_table(client_history)
    global_df = to_round_df(global_rows)

    print("===== Individual Client Results (Mean over Rounds) =====\n")
    print(client_df)
    print("\nFederated Global Fed-GDBD Results\n")
    print_round_table(global_df)
    return client_df, global_df


def run_zerodfl(client_data, CFG=CFG):
    # ============================================
    # Implementation — ZeroDFL (fully decentralised, NO server)
    #   * neighbourhood: fixed ring topology defined once before training; client i talks to i-1 and i+1
    #   * each round: local training -> peer-to-peer gossip of the SHARED encoder only (Eq. 6)
    #   * local adapters, local prompt vectors (lambda = 1) and task heads never leave the hospital
    #   * no global broadcast, no client-to-server upload, no server-side aggregation
    # ============================================

    A = CFG["stage_A_tabular"]
    SEED = 7
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    # fixed ring neighbourhood (static for all rounds)
    NEIGHBOURS = {cid: [CLIENTS[(k - 1) % len(CLIENTS)], CLIENTS[(k + 1) % len(CLIENTS)]]
                  for k, cid in enumerate(CLIENTS)}

    def gossip_round(states):
        """theta_i <- (theta_i + sum_{j in N(i)} theta_j) / (|N(i)| + 1), computed peer-to-peer."""
        return {cid: average_states([states[cid]] + [states[j] for j in NEIGHBOURS[cid]]) for cid in CLIENTS}

    models = {cid: ClientModel(client_data[cid], use_prompt=True,
                               prompt_scale=A["zerodfl_prompt_scale_lambda"]).to(DEVICE) for cid in CLIENTS}
    init = shared_state(models[CLIENTS[0]])
    for m in models.values():
        load_shared(m, init)                       # common initialisation agreed before training
    sizes_test = [len(client_data[cid]["y_test"]) for cid in CLIENTS]

    client_history = {cid: [] for cid in CLIENTS}
    global_rows = []

    for r in range(A["rounds"]):
        # ---- local updates ----
        for cid in CLIENTS:
            spec, model = client_data[cid], models[cid]
            train_local(model, client_loader(spec, bs=A["batch_size"]), A["local_epochs"], A["client_lr"])
            client_history[cid].append(full_metrics(*predict_proba(model, spec)))

        # ---- decentralised gossip on the shared encoder (ring) ----
        mixed = gossip_round({cid: shared_state(models[cid]) for cid in CLIENTS})
        for cid in CLIENTS:
            load_shared(models[cid], mixed[cid])

        # ---- network-level result: every client evaluates its own post-gossip model locally ----
        per_client = [full_metrics(*predict_proba(models[cid], client_data[cid])) for cid in CLIENTS]
        global_rows.append({"Round": r + 1, **pooled_metrics(per_client, sizes_test)})

    client_df = client_mean_table(client_history)
    global_df = to_round_df(global_rows)

    print("===== Individual Client Results (Mean over Rounds) =====\n")
    print(client_df)
    print("\n===== Federated Global ZeroDFL Results =====\n")
    print_round_table(global_df)
    return client_df, global_df


def run_tactfl(client_data, CFG=CFG):
    # ============================================
    # Implementation — TACTFL (server-coordinated, sequential two-stage procedure)
    #   Stage 1: federated temporal contrastive pre-training (InfoNCE, tau = 0.2, no labels)
    #            positives  = second augmented view of the same temporal window
    #            negatives  = all other windows in the mini-batch (anchor and its positive excluded)
    #            server     = similarity-aware weighted aggregation of the SHARED temporal encoder,
    #                         weights computed from each client's contrastive summary vector
    #   Stage 2: shared encoder frozen; each hospital fine-tunes its LOCAL head on 20 % labelled windows
    #            (no joint loss, hence no balancing weight alpha)
    # ============================================

    A = CFG["stage_A_tabular"]
    SEED = 12
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    W, TAU, LAB_FRAC = A["tactfl_window"], A["tactfl_tau"], A["tactfl_label_fraction"]

    # ---- temporal windows of consecutive records (arrival order is preserved by the local split) ----
    def make_windows(spec, split, w=W, stride=W // 2):
        X, E, y = spec[f"X_{split}"], spec[f"E_{split}"], spec[f"y_{split}"]
        starts = np.arange(0, len(y) - w + 1, stride)
        idx = starts[:, None] + np.arange(w)[None, :]
        return torch.tensor(X[idx]), torch.tensor(E[idx]), torch.tensor(y[idx[:, -1]], dtype=torch.long)

    windows = {cid: {s: make_windows(client_data[cid], s) for s in ("train", "test")} for cid in CLIENTS}

    class TemporalEncoder(nn.Module):                     # SHARED
        def __init__(self, d=SHARED_DIM, hid=64, proj=64):
            super().__init__()
            self.gru = nn.GRU(d, hid, batch_first=True)
            self.proj = nn.Sequential(nn.Linear(hid, proj), nn.ReLU(inplace=True), nn.Linear(proj, proj))

        def forward(self, h):                              # h: [B, W, d]
            _, last = self.gru(h)
            return F.normalize(self.proj(last[-1]), dim=1)

    class TACTFLClient(nn.Module):
        def __init__(self, spec):
            super().__init__()
            self.adapter = LocalInputAdapter(spec["X_train"].shape[1], spec["cardinalities"])   # LOCAL
            self.encoder = TemporalEncoder()                                                     # SHARED
            self.head = nn.Linear(64, spec["n_classes"])                                         # LOCAL

        def embed_steps(self, x, e):                        # [B, W, ...] -> [B, W, d]
            B, Wn = x.shape[:2]
            return self.adapter(x.reshape(B * Wn, -1), e.reshape(B * Wn, -1)).reshape(B, Wn, -1)

        def forward(self, x, e):
            return self.head(self.encoder(self.embed_steps(x, e)))

    def augment(h, sigma=0.01, min_keep=0.8):
        """Temporal augmentation on the step embeddings: jitter + random crop (zero-padded)."""
        B, Wn, D = h.shape
        keep = max(int(Wn * min_keep), 1)
        start = np.random.randint(0, Wn - keep + 1)
        mask = torch.zeros(1, Wn, 1, device=h.device)
        mask[:, start:start + keep] = 1.0
        return (h + sigma * torch.randn_like(h)) * mask

    def info_nce(z1, z2, tau=TAU):
        """In-batch InfoNCE: for each anchor the matched view is the positive, the other 2B-2 windows are negatives."""
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = z @ z.t() / tau
        sim.masked_fill_(torch.eye(2 * B, device=z.device, dtype=torch.bool), -9e15)   # exclude the anchor itself
        targets = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
        return F.cross_entropy(sim, targets)

    def similarity_weights(summaries, temp=0.1):
        """Clients whose contrastive summary agrees with the federation receive larger aggregation weight."""
        S = F.normalize(torch.stack(summaries), dim=1)
        sim = S @ S.t()
        agreement = (sim.sum(1) - 1) / (len(summaries) - 1)
        return torch.softmax(agreement / temp, dim=0).numpy()

    def windows_loader(cid, split, bs, shuffle, frac=1.0):
        X, E, y = windows[cid][split]
        n = max(1, int(frac * len(y)))
        return DataLoader(TensorDataset(X[:n], E[:n], y[:n]), batch_size=bs, shuffle=shuffle)

    @torch.no_grad()
    def window_proba(model, cid, split="test"):
        model.eval()
        X, E, y = windows[cid][split]
        probs = [F.softmax(model(X[i:i + 1024].to(DEVICE), E[i:i + 1024].to(DEVICE)), 1).cpu().numpy()
                 for i in range(0, len(y), 1024)]
        return y.numpy(), np.vstack(probs)

    def fit_local_head(model, cid, epochs=10):
        """Stage 2: encoder (and adapter) frozen, only the local head is trained on 20 % labelled windows."""
        model.head.reset_parameters()
        opt = torch.optim.Adam(model.head.parameters(), lr=A["server_head_lr"])
        model.eval(); model.head.train()
        for _ in range(epochs):
            for xb, eb, yb in windows_loader(cid, "train", A["batch_size"], True, LAB_FRAC):
                xb, eb, yb = xb.to(DEVICE), eb.to(DEVICE), yb.to(DEVICE)
                with torch.no_grad():
                    z = model.encoder(model.embed_steps(xb, eb))
                loss = F.cross_entropy(model.head(z), yb)
                opt.zero_grad(); loss.backward(); opt.step()

    models = {cid: TACTFLClient(client_data[cid]).to(DEVICE) for cid in CLIENTS}
    global_shared = shared_state(models[CLIENTS[0]])
    sizes_test = [len(windows[cid]["test"][2]) for cid in CLIENTS]

    client_history = {cid: [] for cid in CLIENTS}
    global_rows, contrastive_curve = [], []

    for r in range(A["rounds"]):
        uploads, summaries, losses = [], [], []
        # ---------- Stage 1: local contrastive training ----------
        for cid in CLIENTS:
            model = models[cid]
            load_shared(model, global_shared)
            params = list(model.adapter.parameters()) + list(model.encoder.parameters())
            opt = torch.optim.Adam(params, lr=A["client_lr"])
            model.train()
            for _ in range(A["local_epochs"]):
                for xb, eb, _ in windows_loader(cid, "train", A["batch_size"], True):
                    h = model.embed_steps(xb.to(DEVICE), eb.to(DEVICE))
                    loss = info_nce(model.encoder(augment(h)), model.encoder(augment(h)))
                    opt.zero_grad(); loss.backward(); opt.step()
                    losses.append(loss.item())
            with torch.no_grad():                                   # contrastive summary (mean embedding)
                xb, eb, _ = next(iter(windows_loader(cid, "train", 512, False)))
                summaries.append(model.encoder(model.embed_steps(xb.to(DEVICE), eb.to(DEVICE))).mean(0).cpu())
            uploads.append(shared_state(model))

            fit_local_head(model, cid)                              # client-level linear probe
            client_history[cid].append(full_metrics(*window_proba(model, cid)))

        # ---------- server: similarity-aware aggregation ----------
        global_shared = average_states(uploads, weights=similarity_weights(summaries))
        contrastive_curve.append(np.mean(losses))

        # ---------- Stage 2 evaluation: frozen global encoder + local heads ----------
        per_client = []
        for cid in CLIENTS:
            load_shared(models[cid], global_shared)
            fit_local_head(models[cid], cid)
            per_client.append(full_metrics(*window_proba(models[cid], cid)))
        global_rows.append({"Round": r + 1, **pooled_metrics(per_client, sizes_test)})

    client_df = client_mean_table(client_history)
    global_df = to_round_df(global_rows)

    print("===== Individual Client Results (Mean over Rounds) =====\n")
    print(client_df)
    print("\n===== Federated Global TACTFL Results =====\n")
    print_round_table(global_df)
    return client_df, global_df


