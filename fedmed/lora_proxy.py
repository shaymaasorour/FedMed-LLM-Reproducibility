"""Stage D: LoRA hybrid tables (Tables 1-3, Table 9) on the five hospital clients of Part 1.

Each hospital keeps its own schema (local input adapter) and label space (local task head); only the LoRA adapters
of the shared backbone (TreeXformer / ClinicalBERT proxy / TFT proxy) are coordinated.
"""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from peft import LoraConfig, get_peft_model
from .config import CFG


def create_client_data(spec, samples_per_client, seed=42):
    """Records of ONE hospital (dense features): 80 % from its train split, 20 % from its test split."""
    rng = np.random.RandomState(seed)
    n_tr = int(0.8 * samples_per_client)
    tr = rng.choice(len(spec["y_train"]), size=min(n_tr, len(spec["y_train"])), replace=False)
    te = rng.choice(len(spec["y_test"]), size=min(samples_per_client - n_tr, len(spec["y_test"])), replace=False)
    as_tensors = lambda X, y: (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return as_tensors(spec["X_train"][tr], spec["y_train"][tr]), as_tensors(spec["X_test"][te], spec["y_test"][te])


def prepare(client_data, cfg=CFG):
    """client_data: {client_id: Part 1 preprocessed client} -> train / test tensors, input dims and class counts."""
    D = cfg["stage_D_lora_proxy"]
    cids = sorted(client_data)
    train, test = map(list, zip(*[create_client_data(client_data[cid], D["samples_per_client"]) for cid in cids]))
    return {"train": train, "test": test, "dims": [X.shape[1] for X, _ in train],
            "classes": [client_data[cid]["n_classes"] for cid in cids]}


class TreeXformer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, x):
        return self.net(x)


class ClinicalBERTProxy(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, 64)

    def forward(self, x):
        return self.encoder(x)


class TFTProxy(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.temporal = nn.GRU(in_dim, 64, batch_first=True)
        self.proj = nn.Linear(64, 64)

    def forward(self, x):
        out, _ = self.temporal(x.unsqueeze(1))
        return self.proj(out[:, -1, :])


BACKBONES = {"tree": TreeXformer, "bert": ClinicalBERTProxy, "tft": TFTProxy}


class HybridModel(nn.Module):
    """SHARED backbone between LOCAL input adapters and LOCAL task heads (one pair per hospital)."""
    def __init__(self, backbone, in_dims, n_classes, input_dim):
        super().__init__()
        self.backbone = backbone
        self.inps = nn.ModuleList([nn.Linear(d, input_dim) for d in in_dims])
        self.heads = nn.ModuleList([nn.Linear(64, k) for k in n_classes])

    def forward(self, x, k=0):
        return self.heads[k](self.backbone(F.relu(self.inps[k](x))))


def apply_lora(model, D):
    lora_cfg = LoraConfig(r=D["lora_r"], lora_alpha=D["lora_scaling"], lora_dropout=D["lora_dropout"], bias="none",
                          target_modules=["net.0", "net.2", "encoder", "proj"])    # SHARED backbone adapters
    model = get_peft_model(model, lora_cfg)
    for n, p in model.named_parameters():                        # LOCAL adapters and task heads (trained, never averaged)
        if ".inps." in n or ".heads." in n:
            p.requires_grad = True
    return model


def train_one_epoch(model, loader, optimizer, k=0):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, k), y)
        loss.backward()
        optimizer.step()


def evaluate(model, X, y, k=0):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X, k), dim=1).numpy()
    preds = probs.argmax(1)
    try:
        auc = (roc_auc_score(y, probs[:, 1]) if probs.shape[1] == 2
               else roc_auc_score(y, probs, multi_class="ovr", labels=list(range(probs.shape[1]))))
    except ValueError:                                           # a class missing from this client's test sample
        auc = np.nan
    return {"Accuracy": accuracy_score(y, preds),
            "Recall": recall_score(y, preds, average="macro", zero_division=0),
            "Precision": precision_score(y, preds, average="macro", zero_division=0),
            "F1": f1_score(y, preds, average="macro", zero_division=0),
            "ROC-AUC": auc}


def federated_train(backbone_class, data, cfg=CFG, rounds=None, aggregation="server"):
    """aggregation='server': FedAvg of LoRA adapters; 'gossip': ring peer-to-peer averaging (ZeroDFL, no server)."""
    D = cfg["stage_D_lora_proxy"]
    n = len(data["train"])
    backbone = backbone_class(D["input_dim"])                    # common initialisation of the shared backbone
    clients = [apply_lora(HybridModel(copy.deepcopy(backbone), [d], [k], D["input_dim"]), D)
               for d, k in zip(data["dims"], data["classes"])]
    init = {name: v.clone() for name, v in clients[0].state_dict().items() if "lora" in name}
    for c in clients[1:]:
        c.load_state_dict(init, strict=False)                    # identical initial adapters on every client
    rounds = rounds or D["rounds"]
    ring = {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}
    snapshots = clients

    for _ in range(rounds):
        for i, model in enumerate(clients):
            loader = DataLoader(TensorDataset(*data["train"][i]), batch_size=D["batch_size"], shuffle=True)
            opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=D["lr"])
            for _ in range(D["local_epochs"]):
                train_one_epoch(model, loader, opt)

        snapshots = [copy.deepcopy(c) for c in clients]           # local states before coordination
        with torch.no_grad():
            lora_names = [name for name, _ in clients[0].named_parameters() if "lora" in name]
            states = [c.state_dict() for c in clients]
            if aggregation == "server":
                avg = {name: sum(s[name] for s in states) / n for name in lora_names}
                mixed = [avg] * n
            else:
                mixed = [{name: (states[i][name] + sum(states[j][name] for j in ring[i])) / (len(ring[i]) + 1)
                          for name in lora_names} for i in range(n)]
            for c, m in zip(clients, mixed):
                for name, v in m.items():
                    c.state_dict()[name].copy_(v)
    return clients, snapshots


def centralized_train(backbone_class, data, cfg=CFG, epochs=5):
    """Pooled multi-task reference: one shared backbone trained on every hospital, each with its own adapter and head."""
    D = cfg["stage_D_lora_proxy"]
    model = apply_lora(HybridModel(backbone_class(D["input_dim"]), data["dims"], data["classes"], D["input_dim"]), D)
    loaders = [DataLoader(TensorDataset(X, y), batch_size=D["batch_size"], shuffle=True) for X, y in data["train"]]
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=D["lr"])
    for _ in range(epochs):
        for k, loader in enumerate(loaders):
            train_one_epoch(model, loader, opt, k)
    return model


def lora_results_table(backbone_class, data, cfg=CFG, rounds=None, aggregation="server"):
    rounds = rounds or cfg["stage_D_lora_proxy"]["rounds"]
    clients, snapshots = federated_train(backbone_class, data, cfg, rounds=rounds, aggregation=aggregation)
    sizes = [len(t[1]) for t in data["test"]]
    weighted = lambda ms: {k: np.average([m[k] for m in ms], weights=sizes) for k in ms[0]}
    fed = [evaluate(c, *data["test"][i]) for i, c in enumerate(clients)]   # shared adapters + local heads
    central = centralized_train(backbone_class, data, cfg, epochs=rounds)
    cols = {"LoRA Federated": weighted(fed),
            "LoRA Centralized": weighted([evaluate(central, *data["test"][i], k=i) for i in range(len(clients))])}
    for i, s in enumerate(snapshots):
        cols[f"LoRA Client-{i + 1}"] = evaluate(s, *data["test"][i])
    table = pd.DataFrame(cols).rename(index={"F1": "F1-score"}).round(4)
    table.index.name = "Metrics"
    return table
