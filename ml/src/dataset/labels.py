from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from .load_manifest import load_manifest

LABEL_TYPES = ("asd_label", "risk_label")
ALLOWED_LABELS = {0, 1}


def make_labels_template(
    manifest_csv: Path,
    out_csv: Path,
    label_type: Literal["asd_label", "risk_label"] = "asd_label",
) -> None:
    m = load_manifest(manifest_csv, require_paths_exist=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    t = pd.DataFrame(
        {
            "child_id": m["child_id"].astype(str),
            "label": pd.Series([pd.NA] * len(m), dtype="Int64"),
            "label_type": [label_type] * len(m),
            "label_source": [pd.NA] * len(m),
            "label_date": [pd.NA] * len(m),
            "notes": [pd.NA] * len(m),
        }
    ).sort_values("child_id")

    t.to_csv(out_csv, index=False)
    print(f"Wrote template: {out_csv}")
    print("Fill label as 0/1 (leave blank if unknown) + label_source.")


def _normalize_labels_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = pd.Index([c.strip() for c in df.columns])

    for c in ["child_id", "label"]:
        if c not in df.columns:
            raise ValueError(f"labels.csv missing required column: '{c}'")

    df["child_id"] = df["child_id"].astype(str).str.strip()
    if (df["child_id"] == "").any():
        bad = df.index[df["child_id"] == ""].tolist()[:10]
        raise ValueError(f"Empty child_id in labels rows (up to 10): {bad}")

    # nullable int
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")

    # Optional columns
    for opt in ["label_type", "label_source", "label_date", "notes"]:
        if opt not in df.columns:
            df[opt] = pd.NA
        df[opt] = df[opt].astype("string").str.strip()

    return df


def validate_and_clean_labels(
    manifest_csv: Path,
    labels_csv: Path,
    out_csv: Path,
    drop_unlabeled: bool = True,
) -> None:
    m = load_manifest(manifest_csv, require_paths_exist=False)
    lbl = _normalize_labels_df(pd.read_csv(labels_csv))

    # Duplicates
    if lbl["child_id"].duplicated().any():
        dups = lbl[lbl["child_id"].duplicated(keep=False)]["child_id"].value_counts()
        raise ValueError(
            "labels.csv has duplicate child_id rows.\n"
            f"Duplicates:\n{dups.to_string()}"
        )

    manifest_ids = set(m["child_id"].tolist())
    label_ids = set(lbl["child_id"].tolist())

    extra = sorted(list(label_ids - manifest_ids))
    if extra:
        raise ValueError(
            "labels.csv contains child_id not present in manifest.csv (up to 20 shown):\n"
            + "\n".join([f"- {x}" for x in extra[:20]])
        )

    missing = sorted(list(manifest_ids - label_ids))
    if missing:
        print(
            f"WARNING: {len(missing)} manifest child_id values are missing from labels.csv "
            f"(up to 20 shown): {missing[:20]}"
        )

    # Label values check
    non_null = lbl["label"].notna()
    bad = lbl.loc[non_null & ~lbl["label"].isin(list(ALLOWED_LABELS)), ["child_id", "label"]]
    if len(bad) > 0:
        raise ValueError(
            "Invalid label values (allowed 0/1 or blank). Examples (up to 20):\n"
            + bad.head(20).to_string(index=False)
        )

    # label_type check (if provided)
    lt_non_null = lbl["label_type"].notna() & (lbl["label_type"] != "")
    bad_lt = lbl.loc[lt_non_null & ~lbl["label_type"].isin(LABEL_TYPES), ["child_id", "label_type"]]
    if len(bad_lt) > 0:
        raise ValueError(
            f"Invalid label_type values (allowed: {LABEL_TYPES}). Examples (up to 20):\n"
            + bad_lt.head(20).to_string(index=False)
        )

    # Recommended: label_source present if label present
    needs_source = lbl["label"].notna() & (lbl["label_source"].isna() | (lbl["label_source"] == ""))
    if needs_source.any():
        ex = lbl.loc[needs_source, ["child_id", "label"]].head(20)
        print(
            "WARNING: Some labeled rows have empty label_source (recommended). Examples (up to 20):\n"
            + ex.to_string(index=False)
        )

    # Align to manifest and optionally drop unlabeled
    merged = m[["child_id"]].merge(lbl, on="child_id", how="left", validate="one_to_one")

    if drop_unlabeled:
        cleaned = merged[merged["label"].isin(list(ALLOWED_LABELS))].copy()
    else:
        cleaned = merged.copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_csv, index=False)

    n_total = len(m)
    n_labeled = int(merged["label"].isin(list(ALLOWED_LABELS)).sum())

    print(f"Manifest children: {n_total}")
    print(f"Labeled children (0/1): {n_labeled}")
    print(f"Wrote: {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Part A: labels workflow")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_t = sub.add_parser("template", help="Create labels.csv template")
    ap_t.add_argument("--manifest", default="data/manifest.csv")
    ap_t.add_argument("--out", default="data/labels.csv")
    ap_t.add_argument("--label_type", default="asd_label", choices=LABEL_TYPES)

    ap_v = sub.add_parser("validate", help="Validate labels.csv -> labels_clean.csv")
    ap_v.add_argument("--manifest", default="data/manifest.csv")
    ap_v.add_argument("--labels", default="data/labels.csv")
    ap_v.add_argument("--out", default="data/derived/labels_clean.csv")
    ap_v.add_argument("--keep_unlabeled", action="store_true")

    args = ap.parse_args()

    if args.cmd == "template":
        make_labels_template(Path(args.manifest), Path(args.out), args.label_type)
    elif args.cmd == "validate":
        validate_and_clean_labels(
            Path(args.manifest),
            Path(args.labels),
            Path(args.out),
            drop_unlabeled=not args.keep_unlabeled,
        )
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
