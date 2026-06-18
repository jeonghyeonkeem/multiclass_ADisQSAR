import os
import re
import glob
import argparse
import pickle
import joblib

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def load_obj(path: str):

    try:
        return joblib.load(path)
    except Exception:
        pass

    with open(path, "rb") as f:
        return pickle.load(f)


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, dtype=str)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str, engine="python")
    if ext == ".csv":
        return pd.read_csv(path, dtype=str)
    raise ValueError("external must be xlsx/tsv/csv")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s).replace("\ufeff", "").strip()).lower()


def find_col(df: pd.DataFrame, name: str) -> str:
    t = _norm(name)
    for c in df.columns:
        if _norm(c) == t:
            return c
    for c in df.columns:
        if t in _norm(c):
            return c
    raise KeyError(f"Column '{name}' not found. Columns={list(df.columns)}")


def parse_class_map(s: str) -> dict[int, int]:
    out = {}
    for item in s.split(","):
        k, v = item.split(":")
        out[int(k.strip())] = int(v.strip())
    return out

def infer_g1(model_path: str) -> str:
    m = re.search(r"Vary_params_results[\\/](\d+)[\\/]", model_path)
    return m.group(1) if m else "g1_unknown"


def infer_ecfp_params(model_path: str) -> tuple[int, int]:
    m = re.search(r"ECFP(\d+)_([0-9]+)bits", model_path, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse ECFP params from model path: {model_path}")
    return int(m.group(1)), int(m.group(2))


def infer_scaler_tag(model_path: str) -> str:
    m = re.search(r"(MinMax|Standard|Robust)", model_path, re.IGNORECASE)
    return m.group(1) if m else "NoScaler"


def infer_model_tag(model_path: str) -> str:
    p = model_path.lower()
    if re.search(r"(?<![a-z])rf(?![a-z])|randomforest", p):
        return "RF"
    if "xgb" in p or "xgboost" in p:
        return "XGB"
    if "svm" in p:
        return "SVM"
    if "mlp" in p:
        return "MLP"
    return "MODEL"


def default_output_name(external_path: str, model_path: str) -> str:
    base = os.path.splitext(os.path.basename(external_path))[0]
    g1 = infer_g1(model_path)
    r, b = infer_ecfp_params(model_path)
    scaler = infer_scaler_tag(model_path)
    mname = infer_model_tag(model_path)
    return f"{base}_{g1}_ECFP{r}({b})_{scaler}_{mname}.xlsx"


def smiles_to_ecfp(smiles: str, radius: int, nbits: int):
    if smiles is None:
        return None
    s = str(smiles).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def is_predictor(obj) -> bool:
    return hasattr(obj, "predict")


def is_scaler(obj) -> bool:
    return hasattr(obj, "transform") and not hasattr(obj, "predict")


def try_unwrap(obj):
    # dict bundle
    if isinstance(obj, dict):
        if "model" in obj:
            return obj["model"], obj.get("scaler", None)
        for k in ["clf", "classifier", "estimator"]:
            if k in obj:
                return obj[k], obj.get("scaler", obj.get("preprocessor", None))

    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        a, b = obj
        if is_predictor(a) and is_scaler(b):
            return a, b
        if is_predictor(b) and is_scaler(a):
            return b, a
    return None, None


def nearby_candidates(seed_path: str):
    seed_path = os.path.abspath(seed_path)
    d = os.path.dirname(seed_path)
    parents = [d, os.path.dirname(d)]
    pats = ["*.pkl", "*.joblib", "*.jl"]

    files = [seed_path]
    for base in parents:
        for pat in pats:
            files.extend(glob.glob(os.path.join(base, pat)))

    seen = set()
    out = []
    for f in files:
        af = os.path.abspath(f)
        if af not in seen and os.path.isfile(af):
            seen.add(af)
            out.append(af)
    return out


def load_predictor_and_scaler(seed_path: str):
    model = None
    scaler = None
    model_src = None
    scaler_src = None

    for p in nearby_candidates(seed_path):
        try:
            obj = load_obj(p)
        except Exception:
            continue

        m, s = try_unwrap(obj)
        if m is not None and is_predictor(m):
            model = m
            model_src = p
            if s is not None and is_scaler(s):
                scaler = s
                scaler_src = p
            break

        if model is None and is_predictor(obj):
            model = obj
            model_src = p
        elif scaler is None and is_scaler(obj):
            scaler = obj
            scaler_src = p

        if model is not None:
            break

    if model is None:
        seed_obj = load_obj(seed_path)
        raise TypeError(
            f"Could not find a predictor (.predict()). Seed type={type(seed_obj)} seed={seed_path}\n"
            "Please point --model to the actual classifier file or keep it in the same folder."
        )

    return model, scaler, model_src, scaler_src


def maybe_make_feature_dataframe(X: np.ndarray, model, nbits: int) -> pd.DataFrame | np.ndarray:

    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) == X.shape[1]:
        return pd.DataFrame(X, columns=list(names))
    # 흔한 케이스: bit_0..bit_{nbits-1}
    cols = [f"bit_{i}" for i in range(nbits)]
    return pd.DataFrame(X, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--smiles_col", default="Smiles")
    ap.add_argument("--class_map", default="0:3,1:2,2:1")
    args = ap.parse_args()

    radius, nbits = infer_ecfp_params(args.model)

    if args.output is None:
        out_name = default_output_name(args.external, args.model)
        args.output = os.path.join(os.path.dirname(args.external) or ".", out_name)

    cmap = parse_class_map(args.class_map)

    df = read_table(args.external)
    smi_col = find_col(df, args.smiles_col)

    X_list, valid_rows, invalid_rows = [], [], []
    for i, smi in enumerate(df[smi_col].tolist()):
        fp = smiles_to_ecfp(smi, radius=radius, nbits=nbits)
        if fp is None:
            invalid_rows.append(i)
        else:
            X_list.append(fp)
            valid_rows.append(i)

    if not X_list:
        raise RuntimeError("No valid SMILES found.")

    X = np.vstack(X_list)

    model, scaler, model_src, scaler_src = load_predictor_and_scaler(args.model)

    X_in = scaler.transform(X) if (scaler is not None and hasattr(scaler, "transform")) else X
    X_for_model = maybe_make_feature_dataframe(X_in, model, nbits)

    pred = model.predict(X_for_model)
    proba = model.predict_proba(X_for_model) if hasattr(model, "predict_proba") else None
    classes = getattr(model, "classes_", np.array([0, 1, 2]))

    df["pred_class"] = ""
    for r, y in zip(valid_rows, pred):
        yi = int(y)
        df.at[r, "pred_class"] = cmap.get(yi, yi)

    if proba is not None:
        for cls in classes:
            df[f"prob_{cmap.get(int(cls), int(cls))}"] = ""
        for r, p in zip(valid_rows, proba):
            for j, cls in enumerate(classes):
                df.at[r, f"prob_{cmap.get(int(cls), int(cls))}"] = float(p[j])

    if invalid_rows:
        df["invalid_smiles_flag"] = ""
        for r in invalid_rows:
            df.at[r, "invalid_smiles_flag"] = "INVALID_SMILES"

    df.to_excel(args.output, index=False)

    print(f"[DONE] Saved: {args.output}")
    print(f"[INFO] inferred fp: radius={radius}, nbits={nbits}")
    print(f"[INFO] model src : {model_src}")
    print(f"[INFO] scaler src: {scaler_src if scaler is not None else 'None'}")
    print(f"[INFO] total rows={len(df)}, predicted={len(valid_rows)}, invalid_smiles={len(invalid_rows)}")


if __name__ == "__main__":
    main()
