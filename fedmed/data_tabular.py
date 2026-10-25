"""Per-hospital preprocessing: 1 dataset = 1 federated client (datasets are never merged)."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATASET_FILES = {
    "Hospital_Emergency": "hospital-emergency-dataset.csv",
    "Triage_History": "hospital-triage-and-patient-history-data.csv",
    "IERAD": "integrated-emergency-response-dataset-ierad.csv",
    "Patient_Data": "hospital-patient-data.csv",
    "Healthcare": "healthcare-dataset.csv",
}

# -----------------------------
# Client roles: each dataset is one hospital with its OWN local task (local prediction head)
# -----------------------------
CLIENT_SPECS = {
    "Hospital_Emergency": {"client": "Client_1", "task": "Operational flow (triage level)",
                           "target": ["Triage Level", "Triage_Level", "ESI Level"]},
    "Triage_History":     {"client": "Client_2", "task": "Admission prediction",
                           "target": ["Admission Outcome", "Admitted", "Disposition", "Outcome"]},
    "IERAD":              {"client": "Client_3", "task": "System-level surge (hospital load)",
                           "target": ["Hospital Load", "Hospital_Load", "Surge Level"]},
    "Patient_Data":       {"client": "Client_4", "task": "Length of stay (LOS band)",
                           "target": ["Stay"]},
    "Healthcare":         {"client": "Client_5", "task": "Robustness site (test result)",
                           "target": ["Test Results"]},
}
ID_COLUMNS = {"case_id", "patientid", "patient_id", "name", "id"}
HIGH_CARD_THRESHOLD = 20          # categorical variables with more levels -> trainable embeddings

norm = lambda c: c.strip().replace(" ", "_").lower()

def clean_df(df):
    df = df.copy()
    df = df.dropna(axis=0, how='all')                # drop all-null rows
    df = df.drop_duplicates()                        # remove duplicates
    df = df.loc[:, ~df.columns.duplicated()]         # remove duplicate cols
    df.columns = [norm(c) for c in df.columns]
    return df

def local_preprocess(df, target_col, seed=42):
    """Local preprocessing executed independently inside one hospital."""
    df = df.dropna(subset=[target_col]).copy()
    y_enc = LabelEncoder()
    y = y_enc.fit_transform(df.pop(target_col).astype(str))

    # identifiers are never used as features
    df = df.drop(columns=[c for c in df.columns if c in ID_COLUMNS
                          or (df[c].dtype == object and df[c].nunique() > 0.9 * len(df))])

    # temporal indexing: timestamps -> hour / day-of-week / month
    for c in [c for c in df.columns if ("date" in c or "time" in c) and df[c].dtype == object]:
        ts = pd.to_datetime(df[c], errors="coerce")
        if ts.notna().mean() > 0.8:
            df[c + "_hour"], df[c + "_dow"], df[c + "_month"] = ts.dt.hour, ts.dt.dayofweek, ts.dt.month
            df = df.drop(columns=c)

    cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in df.columns if c not in cat]
    high = [c for c in cat if df[c].nunique() > HIGH_CARD_THRESHOLD]   # -> embedding ids
    low = [c for c in cat if c not in high]                             # -> label codes

    strat = y if np.bincount(y).min() > 1 else None
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=strat, random_state=seed)
    idx_tr, idx_te = np.sort(idx_tr), np.sort(idx_te)                  # keep arrival order (used by TACTFL)

    dense = pd.concat([df[num], df[low].apply(lambda s: LabelEncoder().fit_transform(s.astype(str)))], axis=1)
    dense = dense.fillna(dense.iloc[idx_tr].median())
    scaler = StandardScaler().fit(dense.iloc[idx_tr])                  # fitted on local TRAIN split only
    X = scaler.transform(dense).astype(np.float32)

    if high:
        E = np.stack([LabelEncoder().fit_transform(df[c].astype(str)) for c in high], axis=1).astype(np.int64)
    else:
        E = np.zeros((len(df), 0), dtype=np.int64)

    return dict(X_train=X[idx_tr], X_test=X[idx_te], E_train=E[idx_tr], E_test=E[idx_te],
                y_train=y[idx_tr], y_test=y[idx_te], n_classes=len(y_enc.classes_),
                classes=list(y_enc.classes_), dense_cols=list(dense.columns),
                high_card_cols=high, cardinalities=[int(df[c].nunique()) for c in high])


def build_clients(data_dir, seed=42):
    """Return {client_id: preprocessed local data} built independently for every hospital."""
    client_data = {}
    for name, spec in CLIENT_SPECS.items():
        path = os.path.join(data_dir, DATASET_FILES[name])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: every hospital client needs its own dataset in {data_dir}")
        d = clean_df(pd.read_csv(path))
        target = next((norm(t) for t in spec["target"] if norm(t) in d.columns), None)
        if target is None:
            raise ValueError(f"no target column {spec['target']} found for {name}")
        client_data[spec["client"]] = {"source": name, "task": spec["task"], "target": target,
                                       **local_preprocess(d, target, seed=seed)}
    return client_data
