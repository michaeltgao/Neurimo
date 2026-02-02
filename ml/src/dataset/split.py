from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, cast

import numpy as np
import pandas as pd


def _validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    required = {"child_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels file missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    out = df.copy()
    out["child_id"] = out["child_id"].astype(str)

    # normalize label to int 0/1
    out["label"] = out["label"].astype(int)
    bad = out[~out["label"].isin([0, 1])]
    if len(bad) > 0:
        raise ValueError(f"labels contain non-binary values. Examples:\n{bad.head(10)}")

    # drop duplicates deterministically
    out = out.drop_duplicates(subset=["child_id"], keep="first").reset_index(drop=True)
    return out


def _stratified_holdout_split(
    child_ids: np.ndarray,
    labels: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Dict[str, List[str]]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")

    rng = np.random.default_rng(seed)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for y in [0, 1]:
        idx = np.where(labels == y)[0]
        rng.shuffle(idx)

        n = idx.size
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        # force remainder to test to avoid off-by-one issues
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        train_ids.extend(child_ids[idx[:n_train]].tolist())
        val_ids.extend(child_ids[idx[n_train : n_train + n_val]].tolist())
        test_ids.extend(child_ids[idx[n_train + n_val : n_train + n_val + n_test]].tolist())

    # final shuffle so class blocks don't remain grouped
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _make_stratified_folds(
    child_ids: np.ndarray,
    labels: np.ndarray,
    k: int,
    seed: int,
) -> Dict[str, List[Dict[str, Union[int, List[str]]]]]:
    """
    Returns:
      {"folds": [{"fold":0,"train":[...],"val":[...]}, ...]}
    Here "val" is the held-out fold. (You can rename to "test" if you prefer.)
    """
    if k < 2:
        raise ValueError("k must be >= 2")

    rng = np.random.default_rng(seed)

    # split indices by class
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    if min(len(idx0), len(idx1)) < k:
        raise ValueError(f"Each class must have at least {k} samples for {k}-fold CV")

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    folds0 = np.array_split(idx0, k)
    folds1 = np.array_split(idx1, k)

    folds_out: List[Dict[str, Union[int, List[str]]]] = []

    for i in range(k):
        val_idx = np.concatenate([folds0[i], folds1[i]])
        train_idx = np.concatenate([np.concatenate([folds0[j] for j in range(k) if j != i]),
                                    np.concatenate([folds1[j] for j in range(k) if j != i])])

        val_ids = child_ids[val_idx].tolist()
        train_ids = child_ids[train_idx].tolist()
        rng.shuffle(val_ids)
        rng.shuffle(train_ids)

        folds_out.append({"fold": i, "train": train_ids, "val": val_ids})

    return {"folds": folds_out}


def _summarize_split(df: pd.DataFrame, split: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """
    Returns counts of total and per-class for each split part.
    """
    by_id = df.set_index("child_id")["label"].to_dict()

    def counts(ids: List[str]) -> Dict[str, int]:
        ys = [by_id[i] for i in ids if i in by_id]
        return {
            "n": len(ids),
            "n_label_0": int(sum(1 for y in ys if y == 0)),
            "n_label_1": int(sum(1 for y in ys if y == 1)),
        }

    return {k: counts(v) for k, v in split.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Create child-level stratified splits.")
    ap.add_argument("--labels", required=True, help="CSV with columns: child_id,label,(optional label_source)")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g., data/derived/splits.json)")

    # holdout config
    ap.add_argument("--mode", choices=["holdout", "kfold"], default="holdout")
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)

    # kfold config
    ap.add_argument("--k", type=int, default=5)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)
    df = _validate_labels(df)

    child_ids = df["child_id"].to_numpy()
    y = df["label"].to_numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict = {
        "labels_path": str(labels_path),
        "n_children": int(len(df)),
        "seed": int(args.seed),
        "mode": args.mode,
    }

    if args.mode == "holdout":
        split = _stratified_holdout_split(
            child_ids=child_ids,
            labels=y,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        payload["split"] = split
        payload["summary"] = _summarize_split(df, split)
    else:
        folds = _make_stratified_folds(
            child_ids=child_ids,
            labels=y,
            k=args.k,
            seed=args.seed,
        )
        payload["folds"] = folds["folds"]
        # fold summaries
        payload["fold_summaries"] = []
        for f in folds["folds"]:
            payload["fold_summaries"].append(
                {"fold": f["fold"], **_summarize_split(df, {"train": cast(List[str], f["train"]), "val": cast(List[str], f["val"])})}
            )

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote splits to: {out_path}")
    if args.mode == "holdout":
        print("Summary:", json.dumps(payload["summary"], indent=2))
    else:
        print(f"Wrote {len(payload['folds'])} folds")


if __name__ == "__main__":
    main()
