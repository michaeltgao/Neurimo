from __future__ import annotations

from pathlib import Path
import pandas as pd

# === YOUR GROUND TRUTH LISTS ===

ASD = {
    "child_000","child_001","child_002","child_003","child_004","child_005","child_006","child_007","child_008","child_009",
    "child_010","child_012","child_014","child_016","child_018","child_020","child_022","child_024","child_026","child_028",
    "child_030","child_032","child_034","child_036","child_038","child_040","child_042","child_044","child_046","child_048",
    "child_050","child_052","child_054","child_056","child_058","child_060","child_062","child_064","child_066","child_068",
    "child_070","child_072","child_074","child_076","child_078","child_080","child_082","child_084","child_086","child_088",
    "child_090","child_092","child_094","child_096","child_098","child_100","child_102","child_104","child_106","child_108",
    "child_110","child_112","child_114","child_116","child_118","child_120","child_122","child_124","child_126","child_128",
    "child_130","child_132","child_134","child_136","child_138","child_140","child_142","child_144","child_146","child_148",
    "child_150","child_152","child_154","child_156","child_158","child_160","child_162","child_164","child_166",
}

NT = {
    "child_011","child_013","child_015","child_017","child_019","child_021","child_023","child_025","child_027","child_029",
    "child_031","child_033","child_035","child_037","child_039","child_041","child_043","child_045","child_047","child_049",
    "child_051","child_053","child_055","child_057","child_059","child_061","child_063","child_065","child_067","child_069",
    "child_071","child_073","child_075","child_077","child_079","child_081","child_083","child_085","child_087","child_089",
    "child_091","child_093","child_095","child_097","child_099","child_101","child_103","child_105","child_107","child_109",
    "child_111","child_113","child_115","child_117","child_119","child_121","child_123","child_125","child_127","child_129",
    "child_131","child_133","child_135","child_137","child_139","child_141","child_143","child_145","child_147","child_149",
    "child_151","child_153","child_155","child_157","child_159","child_161","child_163","child_165",
}


def to_child_key(child_id_value: str) -> str:
    """
    Convert manifest child_id (e.g. '0', '1', '42') to 'child_000', 'child_001', 'child_042'.
    If already 'child_000', returns as-is.
    """
    s = str(child_id_value).strip()
    if s.startswith("child_"):
        return s
    # Allow 'Child000' style if it ever happens
    if s.lower().startswith("child") and s[5:].isdigit():
        return f"child_{int(s[5:]):03d}"
    return f"child_{int(s):03d}"


def main():
    manifest_path = Path("data/manifest.csv")
    out_path = Path("data/labels.csv")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    df = pd.read_csv(manifest_path)
    if "child_id" not in df.columns:
        raise ValueError("manifest.csv must contain 'child_id' column")

    # Sanity: no overlap between sets
    overlap = ASD.intersection(NT)
    if overlap:
        raise ValueError(f"IDs appear in BOTH ASD and NT sets: {sorted(list(overlap))[:20]}")

    df["child_id"] = df["child_id"].astype(str).str.strip()
    df["child_key"] = df["child_id"].apply(to_child_key)

    def label_for_key(k: str):
        if k in ASD:
            return 1
        if k in NT:
            return 0
        return None

    df["label"] = df["child_key"].apply(label_for_key)

    # Report mismatches
    manifest_keys = set(df["child_key"].tolist())
    extra_asd = sorted(list(ASD - manifest_keys))
    extra_nt = sorted(list(NT - manifest_keys))
    if extra_asd:
        print(f"WARNING: {len(extra_asd)} ASD ids not in manifest (up to 20): {extra_asd[:20]}")
    if extra_nt:
        print(f"WARNING: {len(extra_nt)} NT ids not in manifest (up to 20): {extra_nt[:20]}")

    unlabeled = df[df["label"].isna()][["child_id", "child_key"]]
    if len(unlabeled) > 0:
        print(f"WARNING: {len(unlabeled)} manifest children not labeled by ASD/NT lists (up to 20):")
        print(unlabeled.head(20).to_string(index=False))

    labels = df[["child_id", "label"]].copy()
    labels = labels[labels["label"].isin([0, 1])].copy()
    labels["label_source"] = labels["label"].apply(
        lambda x: "synthetic_asd_list" if x == 1 else "synthetic_nt_list"
    )
    labels["notes"] = df.loc[labels.index, "child_key"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print("Counts:")
    print(labels["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
