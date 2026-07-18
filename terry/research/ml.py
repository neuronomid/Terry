"""End-to-end ML research helpers compatible with Jesse 2.5's public API."""

from __future__ import annotations

import csv
import datetime
import os
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, confusion_matrix, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, precision_recall_fscore_support, r2_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def gather_ml_data(config: dict, routes: list[dict], data_routes: list[dict],
                   candles: dict, warmup_candles: dict | None = None,
                   csv_path: str | None = "auto", verbose: bool = True,
                   **backtest_kwargs) -> dict:
    from .backtest import backtest

    result = backtest(
        config, routes, data_routes, candles, warmup_candles=warmup_candles,
        **backtest_kwargs,
    )
    data_points = [point for point in result.get("ml_data", [])
                   if point.get("label") is not None]
    if csv_path == "auto" and routes:
        name = routes[0]["strategy"]
        csv_path = os.path.join("strategies", name, "ml_data", f"{name}_data.csv")
    if csv_path and data_points:
        _write_csv(data_points, csv_path)
    if verbose:
        target = f" → {csv_path}" if csv_path and data_points else ""
        print(f"Collected {len(data_points)} labelled ML samples{target}")
    return {"data_points": data_points, "backtest_metrics": result.get("metrics", {})}


def train_model(data: list[dict], estimator: Any, task: str = "binary",
                test_ratio: float = 0.2, save_to: str | None = None,
                verbose: bool = True, name: str | None = None) -> dict:
    if task not in {"binary", "multiclass", "regression"}:
        raise ValueError("task must be 'binary', 'multiclass', or 'regression'")
    if not data:
        raise ValueError("data is empty — nothing to train on.")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    ordered = sorted(data, key=lambda point: point["time"])
    feature_names = sorted(ordered[0]["features"])
    if not feature_names:
        raise ValueError("data points contain no features")
    for point in ordered:
        if sorted(point["features"]) != feature_names:
            raise ValueError("all data points must use the same feature names")
    X = np.asarray([[point["features"][key] for key in feature_names]
                    for point in ordered], dtype=float)
    raw_labels = [point["label"]["value"] for point in ordered]
    if task == "binary":
        y = np.asarray([1 if _label_is_positive(value) else 0 for value in raw_labels])
    elif task == "multiclass":
        y = np.asarray([int(value) for value in raw_labels])
    else:
        y = np.asarray([float(value) for value in raw_labels], dtype=float)

    split = int(len(X) * (1 - test_ratio))
    if split <= 0 or split >= len(X):
        raise ValueError("test_ratio produces an empty train or test set")
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = clone(estimator)
    model.fit(X_train_scaled, y_train)
    prediction = model.predict(X_test_scaled)

    calibration = None
    class_weights = None
    if task == "binary":
        probabilities = _probabilities(model, X_test_scaled)
        metrics = _classification_metrics(y_test, prediction, probabilities, binary=True)
        calibration = _calibration(y_test, probabilities[:, 1])
        counts = np.bincount(y_train, minlength=2)
        class_weights = ({0: 1.0, 1: float(counts[0]) / counts[1]}
                         if counts[1] else None)
    elif task == "multiclass":
        probabilities = _probabilities(model, X_test_scaled)
        metrics = _classification_metrics(y_test, prediction, probabilities, binary=False,
                                          classes=model.classes_)
    else:
        correlation = spearmanr(y_test, prediction)
        metrics = {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
            "r2": float(r2_score(y_test, prediction)),
            "spearman": float(correlation.statistic),
        }

    importance = _feature_importance(model, feature_names)
    train_test_info = {
        "train_size": len(X_train), "test_size": len(X_test),
        "train_start": _date(ordered[0]["time"]),
        "train_end": _date(ordered[split - 1]["time"]),
        "test_start": _date(ordered[split]["time"]),
        "test_end": _date(ordered[-1]["time"]),
    }
    if save_to:
        os.makedirs(save_to, exist_ok=True)
        joblib.dump(model, os.path.join(save_to, "model.pkl"))
        joblib.dump(scaler, os.path.join(save_to, "scaler.pkl"))
        joblib.dump(importance, os.path.join(save_to, "feature_importance.pkl"))
    if verbose:
        label = f" · {name}" if name else ""
        print(f"Model training{label}: {len(X_train)} train / {len(X_test)} test samples")
        print(metrics)

    output = {
        "model": model, "scaler": scaler, "feature_names": feature_names,
        "metrics": metrics, "feature_importance": importance,
        "feature_impact": [], "train_test_info": train_test_info,
    }
    if task == "binary":
        output.update({"calibration": calibration, "class_weights": class_weights})
    return output


def load_ml_data_csv(path_or_name: str) -> list[dict]:
    if os.sep not in path_or_name and "/" not in path_or_name and not path_or_name.endswith(".csv"):
        path = os.path.join("strategies", path_or_name, "ml_data", f"{path_or_name}_data.csv")
    else:
        path = path_or_name
    if not os.path.exists(path):
        raise FileNotFoundError(f"ML data CSV not found: {path}")
    points = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            names = [key for key in row if key not in {"time", "label_name", "label_value"}]
            points.append({
                "time": int(row["time"]),
                "features": {key: float(row[key]) for key in names},
                "label": {"name": row["label_name"],
                          "value": _parse_label(row["label_value"].strip())},
            })
    return points


def load_ml_model(directory: str) -> dict:
    model_path = os.path.join(directory, "model.pkl")
    scaler_path = os.path.join(directory, "scaler.pkl")
    for path in (model_path, scaler_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file not found: {path}")
    output = {"model": joblib.load(model_path), "scaler": joblib.load(scaler_path)}
    importance_path = os.path.join(directory, "feature_importance.pkl")
    if os.path.exists(importance_path):
        output["feature_importance"] = joblib.load(importance_path)
    return output


def _write_csv(points: list[dict], path: str) -> None:
    features = sorted({key for point in points for key in point["features"]})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "label_name", "label_value", *features])
        for point in sorted(points, key=lambda item: item["time"]):
            writer.writerow([point["time"], point["label"]["name"], point["label"]["value"],
                             *(point["features"].get(name, "") for name in features)])


def _classification_metrics(y_true, y_pred, probabilities, binary, classes=None):
    labels = np.asarray([0, 1]) if binary else np.asarray(classes)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    output = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(), "precision": precision.tolist(),
        "recall": recall.tolist(), "f1": f1.tolist(), "support": support.tolist(),
    }
    try:
        if binary:
            output["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
            output.update(dict(zip(("tn", "fp", "fn", "tp"), map(int, matrix.ravel()))))
        else:
            output["roc_auc_macro"] = float(roc_auc_score(
                y_true, probabilities, multi_class="ovr", average="macro", labels=labels))
            output["classes"] = [_native(value) for value in labels]
    except ValueError:
        output["roc_auc" if binary else "roc_auc_macro"] = float("nan")
    return output


def _probabilities(model, values):
    if not hasattr(model, "predict_proba"):
        raise TypeError("classification estimator must implement predict_proba()")
    return np.asarray(model.predict_proba(values), dtype=float)


def _feature_importance(model, names):
    raw = getattr(model, "feature_importances_", None)
    if raw is None and hasattr(model, "coef_"):
        raw = np.mean(np.abs(np.atleast_2d(model.coef_)), axis=0)
    values = np.asarray(raw if raw is not None else np.zeros(len(names)), dtype=float)
    order = np.argsort(-values)
    return {
        "feature_names": list(names),
        "model_importance": {name: float(values[index]) for index, name in enumerate(names)},
        "consensus_ranks": {names[index]: rank + 1 for rank, index in enumerate(order)},
        "_order": order.tolist(),
    }


def _calibration(y_true, probabilities):
    output = []
    for low, high in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        mask = (probabilities >= low) & (probabilities < high)
        if mask.any():
            output.append({"range": f"[{low:.1f}–{min(high, 1):.1f})", "n": int(mask.sum()),
                           "actual_rate": float(np.mean(y_true[mask])),
                           "expected": (low + min(high, 1)) / 2})
    return output


def _label_is_positive(value):
    if isinstance(value, bool):
        return value
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return str(value).lower() == "true"


def _parse_label(value):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _date(timestamp):
    return datetime.datetime.fromtimestamp(int(timestamp), datetime.UTC).strftime("%Y-%m-%d")


def _native(value):
    return value.item() if isinstance(value, np.generic) else value
