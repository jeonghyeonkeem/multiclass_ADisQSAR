import time
import logging
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from joblib import dump
from Admodule import Utils

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


logger = logging.getLogger(__name__)


def _ensure_int_labels(y: pd.Series) -> np.ndarray:
    return pd.Series(y).astype(int).to_numpy()


def multiclass_auc_ovr_macro(y_true, y_proba) -> float:

    y_true = _ensure_int_labels(y_true)
    if y_proba is None:
        return 0.0
    y_proba = np.asarray(y_proba)

    classes = np.unique(y_true)
    if y_proba.ndim != 2:
        return 0.0
    if y_proba.shape[1] != len(classes):
        return 0.0

    y_bin = label_binarize(y_true, classes=classes)
    try:
        return float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        return 0.0


def get_auc_per_class(y_true, y_proba, all_classes) -> dict:
    y_true = _ensure_int_labels(y_true)
    y_proba = None if y_proba is None else np.asarray(y_proba)

    out = {}
    if y_proba is None or y_proba.ndim != 2 or y_proba.shape[1] != len(all_classes):
        for c in all_classes:
            out[f"AUC_class{c}"] = np.nan
        return out

    # one-vs-rest AUC
    for idx, c in enumerate(all_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            out[f"AUC_class{c}"] = np.nan
            continue
        try:
            out[f"AUC_class{c}"] = float(roc_auc_score(y_bin, y_proba[:, idx]))
        except Exception:
            out[f"AUC_class{c}"] = np.nan
    return out


def GridSearchRUN(model, parameters, X_train, y_train, cores=4):
    """
      - accuracy
      - precision_macro
      - recall_macro
      - f1_macro
      - auc_ovr_macro (needs_proba)
    """
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "auc_ovr_macro": make_scorer(multiclass_auc_ovr_macro, needs_proba=True),
    }

    grid_model = GridSearchCV(
        model,
        param_grid=parameters,
        scoring=scoring,
        refit="f1_macro",
        cv=10,
        n_jobs=cores,
        error_score=0,
        verbose=10,
    )
    grid_model.fit(X_train, y_train)

    result = pd.DataFrame(grid_model.cv_results_["params"])
    result["mean_valid_accuracy"] = grid_model.cv_results_["mean_test_accuracy"]
    result["mean_valid_precision_macro"] = grid_model.cv_results_["mean_test_precision_macro"]
    result["mean_valid_recall_macro"] = grid_model.cv_results_["mean_test_recall_macro"]
    result["mean_valid_f1_macro"] = grid_model.cv_results_["mean_test_f1_macro"]
    result["mean_valid_auc_ovr_macro"] = grid_model.cv_results_["mean_test_auc_ovr_macro"]

    result.sort_values(by="mean_valid_f1_macro", ascending=False, inplace=True)
    logger.info(f"GridSearchCV results :\n{result[result.columns[-5:]]}")
    return grid_model, result


def result_scoring(md, df, cols, name, out, all_classes):
    X = df[cols]
    y = _ensure_int_labels(df["AD"])

    y_pred = md.predict(X)

    y_proba = None
    if hasattr(md, "predict_proba"):
        try:
            y_proba = md.predict_proba(X)
        except Exception:
            y_proba = None

    df = df.copy()
    df["Pred"] = y_pred
    df["Match"] = (df["AD"].astype(int).to_numpy() == y_pred).astype(int)
    Utils.save(df[["Compound_ID", "AD", "Pred", "Match"]], out, custom=f"{name}_prediction_log")

    acc = float(accuracy_score(y, y_pred))
    prec_macro = float(precision_score(y, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y, y_pred, average="macro", zero_division=0))


    auc_macro = np.nan
    if y_proba is not None:
        y_bin = label_binarize(y, classes=all_classes)
        if y_proba.ndim == 2 and y_proba.shape[1] == len(all_classes):
            try:
                auc_macro = float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"))
            except Exception:
                auc_macro = np.nan

    # confusion matrix
    cm = confusion_matrix(y, y_pred, labels=all_classes)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in all_classes], columns=[f"pred_{c}" for c in all_classes])
    Utils.save(cm_df, out, custom=f"{name}_confusion_matrix")

    auc_per_class = get_auc_per_class(y, y_proba, all_classes)

    result = {
        "Data": name,
        "ACC": round(acc, 4),
        "Precision_macro": round(prec_macro, 4),
        "Recall_macro": round(rec_macro, 4),
        "F1_macro": round(f1_macro, 4),
        "AUC_ovr_macro": (round(auc_macro, 4) if isinstance(auc_macro, (float, np.floating)) and not np.isnan(auc_macro) else np.nan),
    }

    for k, v in auc_per_class.items():
        result[k] = (round(v, 4) if isinstance(v, (float, np.floating)) and not np.isnan(v) else np.nan)

    logger.info(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ADis QSAR model (multiclass)")
    parser.add_argument("-train", "--train", required=True, help="Train data")
    parser.add_argument("-valid", "--valid", required=True, help="Valid data")
    parser.add_argument("-test", "--test", default=False, help="Test data")
    parser.add_argument("-o", "--output", type=str, required=True, help="Set your output path")
    parser.add_argument("-m", "--model", type=str, default="RF", help="Set your model type")
    parser.add_argument("-core", "--num_cores", type=int, default=2, help="Set the number of CPU cores to use")
    args = parser.parse_args()

    warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

    # path
    path_train = Path(args.train)
    path_valid = Path(args.valid)
    file_name = path_train.stem.split("_")[0]

    path_output = Path(args.output) / f"{file_name}_model" / args.model
    path_output.mkdir(parents=True, exist_ok=True)

    # log
    Utils.set_log(path_output, "model.log")

    start = time.time()
    n_cores = Utils.set_cores(args.num_cores)

    logger.info(f"Train data : {path_train}")
    logger.info(f"Valid data : {path_valid}")
    logger.info(f"Output path : {path_output}")
    logger.info(f"Model type : {args.model}")
    logger.info(f"Use cores : {n_cores}")

    train = pd.read_csv(path_train, sep="\t")
    valid = pd.read_csv(path_valid, sep="\t")
    xcols = [x for x in train.columns if x.startswith("f_")]

    y_train = _ensure_int_labels(train["AD"])
    all_classes = sorted(np.unique(y_train).tolist())
    n_classes = len(all_classes)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes in train AD. Got: {all_classes}")

    logger.info(f"Detected classes: {all_classes} (n_classes={n_classes})")

    # model init
    if args.model.upper() == "SVM":
        model = SVC(random_state=42, probability=True)
        parameters = {
            "kernel": ["linear", "rbf"],
            "C": [0.01, 0.1, 1, 10, 100, 1000],
            "gamma": [0.01, 0.1, 1, 100, 1000],
            "class_weight": [None, 'balanced']
        }

    elif args.model.upper() == "RF":
        model = RandomForestClassifier(class_weight="balanced", random_state=42)
        parameters = {
            "max_depth": [6, 8, 10, 12],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [2, 4, 8, 10],
            "min_samples_split": [2, 4, 8, 10],
            "n_estimators": [200, 600, 1000],
        }

    elif args.model.upper() == "XGB":
        model = xgb.XGBClassifier(
            n_jobs=0,
            seed=42,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
        )
        parameters = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.2],
            "n_estimators": [200, 600, 1000],
            "gamma": [0, 1000],
            "min_child_weight": [1, 3],
            "subsample": [0.5, 1],
            "colsample_bytree": [0.5, 1],
        }


    elif args.model.upper() == "MLP":
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        model = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('mlp', MLPClassifier(random_state=42))
            ])
        parameters = {
            "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
            "mlp__learning_rate": ["constant", "invscaling", "adaptive"],
            "mlp__activation": ["relu", "identity", "logistic", "tanh"],
            "mlp__solver": ["adam", "sgd", "lbfgs"],
            "mlp__alpha": [0.1, 0.01, 0.001],
            "mlp__max_iter": [500, 1000],
            "mlp__early_stopping": [True],
            }

    else:
        raise ValueError(f"{args.model} model type can not use.")

    logger.info(f"Set parameters : {parameters}")

    # start learning
    grid, train_log = GridSearchRUN(model, parameters, train[xcols], y_train, cores=n_cores)
    grid_model = grid.best_estimator_
    logger.info(f"Best model :\n{grid_model}")

    # scoring
    base_out = path_output / f"{file_name}_{args.model}"
    train_score = result_scoring(grid_model, train, xcols, "train", out=base_out, all_classes=all_classes)
    valid_score = result_scoring(grid_model, valid, xcols, "valid", out=base_out, all_classes=all_classes)

    if args.test:
        path_test = Path(args.test)
        test = pd.read_csv(path_test, sep="\t")
        test_score = result_scoring(grid_model, test, xcols, "test", out=base_out, all_classes=all_classes)
        total_score = pd.DataFrame([train_score, valid_score, test_score])
    else:
        total_score = pd.DataFrame([train_score, valid_score])

    # save model + logs
    dump(grid_model, path_output / f"{file_name}_{args.model}_model.pkl")
    Utils.save(train_log, path_output / f"{file_name}_{args.model}_model", custom="training_log")
    Utils.save(total_score, path_output / f"{file_name}_{args.model}_model", custom="score_log")

    runtime = time.time() - start
    logger.info(f"Time : {runtime}")

