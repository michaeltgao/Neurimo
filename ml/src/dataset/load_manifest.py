from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

REQUIRED_COLS = ["child_id"]
VIDEO_PATH_COLS = ["joint_attention_path", "imitation_path", "free_play_path"]


def load_manifest(manifest_path: Path, require_paths_exist: bool = True) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    df.columns = pd.Index([c.strip() for c in df.columns])

    # Required column(s)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"manifest.csv missing required column: '{c}'")

    # Normalize IDs
    df["child_id"] = df["child_id"].astype(str).str.strip()
    if (df["child_id"] == "").any():
        bad = df.index[df["child_id"] == ""].tolist()[:10]
        raise ValueError(f"Empty child_id found in rows (up to 10): {bad}")

    # One row per child (v1)
    if df["child_id"].duplicated().any():
        dups = df[df["child_id"].duplicated(keep=False)]["child_id"].value_counts()
        raise ValueError(
            "manifest.csv must have one row per child_id.\n"
            f"Duplicates:\n{dups.to_string()}"
        )

    # Optional metadata
    if "age_bucket" in df.columns:
        df["age_bucket"] = df["age_bucket"].astype(str).str.strip()

    # Path checks (only check columns that exist)
    if require_paths_exist:
        for col in VIDEO_PATH_COLS:
            if col not in df.columns:
                continue
            missing = ~df[col].astype(str).apply(lambda p: Path(p).exists())
            if missing.any():
                ex = df.loc[missing, ["child_id", col]].head(20)
                raise FileNotFoundError(
                    f"Missing files referenced in column '{col}' (up to 20 shown):\n{ex.to_string(index=False)}"
                )

    return df


def main():
    ap = argparse.ArgumentParser(description="Validate + clean manifest.csv")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="data/derived/manifest_clean.csv")
    ap.add_argument("--skip_path_check", action="store_true")
    args = ap.parse_args()

    df = load_manifest(Path(args.manifest), require_paths_exist=not args.skip_path_check)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"OK: {len(df)} children")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
