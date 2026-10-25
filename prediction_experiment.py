"""External prediction on dataset/prediction/Additional file 7.csv.

Applies the five base models, majority voting, and SpecEn-Moll. Feature
construction matches utils.data_preprocessing.preprocess: RDKit text
features plus 2-D spectral embeddings; LabelEncoder and StandardScaler
are fit on dataset/train only; unseen object-column values map to -1.

    python prediction_experiment.py
    python prediction_experiment.py --limit 300
"""
import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PREDICTION_DIR = os.path.join(BASE, "dataset", "prediction")
SRC_CSV = os.path.join(PREDICTION_DIR, "Additional file 7.csv")
TRAIN_FEATURE_XLSX = os.path.join(BASE, "dataset", "train", "text_feature.xlsx")
THETA_DIR = os.path.join(BASE, "theta", "specen_moll")
RESULT_DIR = os.path.join(BASE, "results", "prediction")
INTERMEDIATE_DIR = os.path.join(RESULT_DIR, "intermediate")
DIAG_DIR = os.path.join(RESULT_DIR, "diagnostics")

K = 2
UNSEEN_CODE = -1
BASE_MODELS = ["RF", "SVM", "XGB", "LR", "KNN"]
SCALED_MODELS = ["SVM", "LR", "KNN"]

OBJECT_COLS = [
    "smiles", "Mol", "MolFormula", "InChI", "InChIKey",
    "RDKFingerprint", "TopologicalFP", "MACCS", "AtomPairsFP",
    "TopologicalTorsionsFP", "MorganFP_2", "MorganFP_3", "molprint2D",
]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def step1_build_smiles_table(limit=None):
    """Keep RDKit-valid SMILES so graphs stay aligned with the table."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    df = pd.read_csv(SRC_CSV, low_memory=False)
    if "SMILES" not in df.columns:
        raise ValueError(f"{SRC_CSV} has no SMILES column")

    rows, dropped = [], []
    for i, s in enumerate(df["SMILES"]):
        if not isinstance(s, str) or not s.strip():
            dropped.append((i, s, "empty"))
            continue
        if Chem.MolFromSmiles(s) is None:
            dropped.append((i, s, "invalid SMILES"))
            continue
        rows.append({
            "smiles": s,
            "Compound_CID": df["Compound_CID"].iloc[i],
            "Name": df["Name"].iloc[i],
        })

    out = pd.DataFrame(rows).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    if limit:
        out = out.head(limit)

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    out.to_excel(os.path.join(INTERMEDIATE_DIR, "prediction_smiles.xlsx"), index=False)
    log(f"[step1] {len(df)} rows -> {len(out)} unique valid SMILES, dropped {len(dropped)}")
    for i, s, why in dropped:
        log(f"        dropped row {i} ({why}): {str(s)[:70]}")
    return out


def step2_extract_text_features():
    """RDKit descriptors and fingerprints aligned with dataset/train/text_feature.xlsx."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit.Chem.inchi import MolToInchi, MolToInchiKey

    smiles_xlsx = os.path.join(INTERMEDIATE_DIR, "prediction_smiles.xlsx")
    df = pd.read_excel(smiles_xlsx)

    descriptor_names = [d[0] for d in Descriptors._descList]
    columns = ["Mol", "MolFormula", "InChI", "InChIKey"] + descriptor_names + [
        "RDKFingerprint", "TopologicalFP", "MACCS", "AtomPairsFP",
        "TopologicalTorsionsFP", "MorganFP_2", "MorganFP_3", "molprint2D",
    ]

    results = []
    for _, row in df.iterrows():
        smiles = row["smiles"]
        try:
            mol = Chem.MolFromSmiles(smiles)
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
            inchi = MolToInchi(mol)
            inchi_key = MolToInchiKey(mol)
            descriptor_values = [Descriptors.__dict__[n](mol) for n in descriptor_names]
            rdk_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2).ToBitString()
            topo_fp = AllChem.RDKFingerprint(mol).ToBitString()
            maccs_fp = MACCSkeys.GenMACCSKeys(mol).ToBitString()
            atom_pairs_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol).ToBitString()
            topo_torsions_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol).ToBitString()
            morgan_fp_2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2).ToBitString()
            morgan_fp_3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3).ToBitString()
            molprint2D = AllChem.GetMorganFingerprint(mol, radius=2, useFeatures=True).GetNonzeroElements()
            results.append(
                [mol, CalcMolFormula(mol), inchi, inchi_key]
                + descriptor_values
                + [
                    rdk_fp, topo_fp, maccs_fp, atom_pairs_fp, topo_torsions_fp,
                    morgan_fp_2, morgan_fp_3,
                    ",".join(f"{k}:{v}" for k, v in molprint2D.items()),
                ]
            )
        except Exception as e:
            log(f"[step2] feature extraction failed {smiles}: {e}")
            results.append([np.nan] * len(columns))

    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results, columns=columns)], axis=1)
    out = os.path.join(INTERMEDIATE_DIR, "prediction_text_feature.xlsx")
    df.to_excel(out, index=False)
    log(f"[step2] feature table shape={df.shape} -> {out}")
    return df


def step3_build_graph_and_embedding():
    """Build molecular graphs and 2-D spectral embeddings (feature_0, feature_1)."""
    from utils import pc_gnn, spectral_embedding

    smiles_xlsx = os.path.join(INTERMEDIATE_DIR, "prediction_smiles.xlsx")
    graph_pt = os.path.join(INTERMEDIATE_DIR, "prediction_pc-gnn.pt")
    pc_gnn.get_pc_gnn_1(smiles_xlsx, graph_pt)
    emb = spectral_embedding.get_embedding(graph_pt, smiles_xlsx, K)
    log(f"[step3] embedding shape={emb.shape}")
    return emb


def training_model_columns():
    """Training feature columns excluding the label."""
    train_cols = list(pd.read_excel(TRAIN_FEATURE_XLSX).columns)
    return [c for c in train_cols if c != "lable"]


def step4_build_feature_matrix():
    """Fit scaler and LabelEncoder on dataset/train; apply to the prediction set."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    tf = pd.read_excel(TRAIN_FEATURE_XLSX)
    X = tf.drop(columns=["lable"]).dropna()
    encoders = {}
    for col in OBJECT_COLS:
        if col in X.columns:
            le = LabelEncoder().fit(X[col])
            encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            X[col] = le.transform(X[col])
    scaler = StandardScaler().fit(X)
    log(f"[step4] scaler fit on train {X.shape[0]} x {X.shape[1]}")

    model_cols = training_model_columns()
    feature_name = [f"feature_{i}" for i in range(K)]
    df = pd.read_excel(os.path.join(INTERMEDIATE_DIR, "prediction_text_feature.xlsx"))
    emb = step3_build_graph_and_embedding()
    df = pd.concat([df, pd.DataFrame(emb, columns=feature_name)], axis=1)
    df = df.dropna().reset_index(drop=True)

    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        raise KeyError(f"prediction table missing training columns: {missing}")
    extra = [c for c in df.columns if c not in model_cols and c not in feature_name]
    if extra:
        log(f"[step4] extra columns ignored: {extra}")
    model_view = df[model_cols + feature_name].copy()

    unseen_cells = 0
    for col in OBJECT_COLS:
        if col not in model_view.columns:
            continue
        mapping = encoders.get(col, {})
        model_view[col] = model_view[col].map(lambda v: mapping.get(v, UNSEEN_CODE))
        unseen_cells += int((model_view[col] == UNSEEN_CODE).sum())

    text = model_view[model_cols]
    graph = model_view[feature_name]
    X_scaled = np.concatenate((scaler.transform(text), graph.values), axis=1)
    X_raw = model_view
    log(f"[step4] scaled={X_scaled.shape} raw={X_raw.shape} unseen cells={unseen_cells}")
    return df, X_scaled, X_raw, model_cols, scaler, encoders


def step5_predict(df, X_scaled, X_raw):
    """Write base-model, voting, and SpecEn-Moll predictions under results/prediction/."""
    import joblib

    out = pd.DataFrame({"smiles": df["smiles"].values})
    if "Compound_CID" in df.columns:
        out.insert(0, "Compound_CID", df["Compound_CID"].values)
    if "Name" in df.columns:
        out.insert(1, "Name", df["Name"].values)

    cache = np.zeros(len(X_raw), dtype=np.int32)
    proba = {}
    for name in BASE_MODELS:
        grid = joblib.load(os.path.join(THETA_DIR, f"{name}_model.pkl"))
        est = grid.best_estimator_ if hasattr(grid, "best_estimator_") else grid
        if name in SCALED_MODELS:
            y_pred = est.predict(X_scaled)
            p = est.predict_proba(X_scaled)[:, 1]
        else:
            y_pred = est.predict(X_raw)
            p = est.predict_proba(X_raw)[:, 1]
        out[f"{name}_pred"] = y_pred
        out[f"{name}_proba"] = p
        proba[name] = p
        cache += y_pred

        d = os.path.join(RESULT_DIR, "base_models", name)
        os.makedirs(d, exist_ok=True)
        out[["Compound_CID", "Name", "smiles", f"{name}_pred", f"{name}_proba"]].to_excel(
            os.path.join(d, "prediction.xlsx"), index=False
        )
        log(f"[step5] {name}: positive rate={y_pred.mean():.4f} n={int(y_pred.sum())}/{len(y_pred)}")
    out["VOTING_pred"] = (cache >= 4).astype(np.int32)

    try:
        from models import specen_moll
        prior = np.column_stack([proba[n] for n in BASE_MODELS]).astype(np.float32)
        Xj = np.hstack((X_scaled, prior))
        specen = specen_moll.SpecEnMoll(Xj.shape[1], 5)
        specen.load_weights(os.path.join(THETA_DIR, "SpecEn-Moll.ckpt"))
        out["SpecEn_Moll_pred"] = specen.predict1(Xj, prior, 32)
    except Exception as e:
        log(f"[step5] SpecEn-Moll skipped: {e}")
        out["SpecEn_Moll_pred"] = np.nan

    for agg, col in [("voting", "VOTING_pred"), ("SpecEn-Moll", "SpecEn_Moll_pred")]:
        d = os.path.join(RESULT_DIR, "ensemble", agg)
        os.makedirs(d, exist_ok=True)
        out[["Compound_CID", "Name", "smiles", col] + [f"{n}_pred" for n in BASE_MODELS]].to_excel(
            os.path.join(d, "prediction.xlsx"), index=False
        )
        log(f"[step5] {agg}: positive rate={out[col].mean():.4f}")
    return out


def step6_diagnostics(df, out, model_cols, scaler, encoders):
    """Applicability domain, unseen object-column rate, and inter-model agreement."""
    tf = pd.read_excel(TRAIN_FEATURE_XLSX)
    num_cols = [c for c in model_cols if c not in OBJECT_COLS]

    X_train = tf.drop(columns=["lable"]).dropna()
    for c in OBJECT_COLS:
        if c in X_train.columns:
            X_train[c] = X_train[c].map(lambda v: encoders.get(c, {}).get(v, UNSEEN_CODE))
    te_path = os.path.join(BASE, "dataset", "test", "text_feature.xlsx")
    X_test = pd.read_excel(te_path).drop(columns=["lable"]).dropna()
    for c in OBJECT_COLS:
        if c in X_test.columns:
            X_test[c] = X_test[c].map(lambda v: encoders.get(c, {}).get(v, UNSEEN_CODE))

    num_idx = [model_cols.index(c) for c in num_cols]
    Tr = scaler.transform(X_train)
    Te = scaler.transform(X_test)
    lo, hi = Tr[:, num_idx].min(axis=0), Tr[:, num_idx].max(axis=0)
    base_oob = (Te[:, num_idx] < lo) | (Te[:, num_idx] > hi)

    T = df[model_cols].copy()
    for c in OBJECT_COLS:
        if c in T.columns:
            T[c] = T[c].map(lambda v: encoders.get(c, {}).get(v, UNSEEN_CODE))
    Ts = scaler.transform(T)
    oob = (Ts[:, num_idx] < lo) | (Ts[:, num_idx] > hi)

    oob_df = pd.DataFrame({
        "feature": num_cols,
        "oob_rate": oob.mean(axis=0),
        "train_scaled_min": lo,
        "train_scaled_max": hi,
        "prediction_scaled_min": Ts[:, num_idx].min(axis=0),
        "prediction_scaled_max": Ts[:, num_idx].max(axis=0),
    }).sort_values("oob_rate", ascending=False)

    P = np.column_stack([out[f"{n}_pred"].values for n in BASE_MODELS])
    votes = P.sum(axis=1)
    agree = pd.DataFrame({
        "smiles": out["smiles"].values,
        "Compound_CID": out["Compound_CID"].values,
        **{f"{n}_pred": P[:, i] for i, n in enumerate(BASE_MODELS)},
        "votes": votes,
        "unanimous": (P == P[:, [0]]).all(axis=1),
    })

    os.makedirs(DIAG_DIR, exist_ok=True)
    oob_df.to_csv(os.path.join(DIAG_DIR, "applicability_domain.csv"), index=False)
    agree.to_csv(os.path.join(DIAG_DIR, "model_agreement.csv"), index=False)

    unseen_cells = int((T[OBJECT_COLS] == UNSEEN_CODE).sum().sum())
    vote_dist = {int(v): int(c) for v, c in zip(*np.unique(votes, return_counts=True))}
    summary = {
        "n_compounds": int(len(out)),
        "applicability_domain": {
            "oob_value_rate": float(oob.mean()),
            "rows_with_any_oob_rate": float(oob.any(axis=1).mean()),
            "train_holdout_baseline_oob_rate": float(base_oob.mean()),
            "top10_oob_features": oob_df.head(10)["feature"].tolist(),
        },
        "object_columns": {
            "n_columns": len(OBJECT_COLS),
            "n_cells": int(len(T) * len(OBJECT_COLS)),
            "n_unseen_cells": unseen_cells,
            "unseen_rate": float(unseen_cells / (len(T) * len(OBJECT_COLS))),
        },
        "agreement": {
            "unanimous_rate": float(agree["unanimous"].mean()),
            "mean_agreement_with_first": float(np.mean(P == P[:, [0]])),
            "vote_distribution": vote_dist,
        },
        "positive_rate": {c: float(out[c].mean()) for c in out.columns if c.endswith("_pred")},
    }
    with open(os.path.join(DIAG_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("=" * 60)
    log("unlabeled diagnostics")
    log(
        f"  AD: {len(num_cols)} numeric descriptors, OOB value rate {oob.mean():.2%}; "
        f"{oob.any(axis=1).mean():.1%} of molecules have at least one OOB descriptor"
    )
    log(f"      hold-out vs train baseline OOB rate {base_oob.mean():.2%}")
    log(f"  object columns: {len(OBJECT_COLS)} x {len(T)} cells, {unseen_cells} unseen in train")
    log(f"  agreement: unanimous {agree['unanimous'].mean():.1%}; vote dist {vote_dist}")
    log("=" * 60)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process the first N rows")
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    log(f"output directory: {RESULT_DIR}")

    if args.limit:
        log(f"limit={args.limit}")
    step1_build_smiles_table(args.limit)
    step2_extract_text_features()

    df, X_scaled, X_raw, model_cols, scaler, encoders = step4_build_feature_matrix()
    out = step5_predict(df, X_scaled, X_raw)
    step6_diagnostics(df, out, model_cols, scaler, encoders)

    main_xlsx = os.path.join(RESULT_DIR, "prediction.xlsx")
    out.to_excel(main_xlsx, index=False)
    log(f"wrote {main_xlsx} shape={out.shape}")


if __name__ == "__main__":
    main()
