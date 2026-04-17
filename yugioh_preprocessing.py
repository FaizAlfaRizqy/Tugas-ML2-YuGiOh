import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


EFFECT_TAG_PATTERNS: Dict[str, str] = {
    "search": r"\b(add|search)\b.*\b(deck|hand)\b|\badd 1\b",
    "negate": r"\bnegate\b",
    "destroy": r"\bdestroy\b",
    "banish": r"\bbanish\b",
    "draw": r"\bdraw\b",
    "bounce": r"\breturn\b.*\b(hand)\b|\bshuffle\b.*\b(deck)\b",
    "special_summon": r"\bspecial summon\b",
    "send_gy": r"\bsend\b.*\bgraveyard\b|\bGY\b",
    "recycle": r"\badd\b.*\bgraveyard\b|\bshuffle\b.*\bgraveyard\b",
    "lock": r"\bcannot\b|\bneither player can\b",
}

FIRST_TURN_PATTERNS: List[str] = [
    r"\badd\b.*\bdeck\b",
    r"\bsearch\b",
    r"\bspecial summon\b.*\bfrom your deck\b",
    r"\bsend\b.*\bfrom your deck to (the )?graveyard\b",
    r"\bdraw\b",
    r"\bset 1\b",
    r"\bfusion summon\b|\bsynchro summon\b|\bxyz summon\b|\blink summon\b",
]

SECOND_TURN_PATTERNS: List[str] = [
    r"\bwhen your opponent\b",
    r"\bduring your opponent'?s turn\b",
    r"\bwhen an opponent'?s monster declares an attack\b",
    r"\bquick effect\b",
    r"\bnegate\b",
    r"\bdestroy\b.*\bopponent\b",
    r"\bbanish\b.*\bopponent\b",
    r"\bcannot activate cards or effects\b",
    r"\bbattle phase\b",
]


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resolve_csv_path(csv_arg: str | None) -> Path:
    if csv_arg:
        path = Path(csv_arg)
        if not path.exists():
            raise FileNotFoundError(f"CSV tidak ditemukan: {path}")
        return path

    candidates = sorted(Path.cwd().glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("Tidak ada file CSV di folder aktif. Gunakan --csv.")

    # Prefer raw source CSV and avoid using already preprocessed output as input.
    preferred = [p for p in candidates if "preprocessed" not in p.name.lower()]
    if preferred:
        candidates = preferred

    selected = candidates[0]
    print(f"[INFO] Argumen --csv tidak diketik, jadi file dipilih otomatis: {selected}")
    return selected


def resolve_description_col(df: pd.DataFrame, desc_col: str) -> str:
    if desc_col in df.columns:
        return desc_col
    if "description" in df.columns:
        return "description"
    if "desc" in df.columns:
        return "desc"
    raise ValueError(f"Kolom deskripsi tidak ditemukan. Kolom tersedia: {list(df.columns)}")


def extract_effect_tags(text: str) -> List[str]:
    text = "" if pd.isna(text) else str(text).lower()
    tags = [tag for tag, pattern in EFFECT_TAG_PATTERNS.items() if re.search(pattern, text)]
    return tags if tags else ["other"]


def heuristic_turn_label(text: str, card_type: str = "", sub_type: str = "") -> Tuple[int, str, int, int]:
    text = "" if pd.isna(text) else str(text).lower()
    card_type = "" if pd.isna(card_type) else str(card_type).lower()
    sub_type = "" if pd.isna(sub_type) else str(sub_type).lower()

    first_score = sum(1 for p in FIRST_TURN_PATTERNS if re.search(p, text))
    second_score = sum(1 for p in SECOND_TURN_PATTERNS if re.search(p, text))

    if "trap" in card_type:
        second_score += 1
    if "quick-play" in sub_type:
        second_score += 1

    if first_score > second_score:
        return 0, "heuristic_first", first_score, second_score
    if second_score > first_score:
        return 1, "heuristic_second", first_score, second_score

    fallback = 1 if "trap" in card_type else 0
    return fallback, "heuristic_tie_break", first_score, second_score


def build_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col in df.columns:
        candidate = pd.to_numeric(df[target_col], errors="coerce")
        valid = candidate.dropna().isin([0, 1]).all()
        if valid and candidate.notna().sum() > 0:
            df["target"] = candidate.astype("Int64")
            df["target_source"] = "existing"
            return df

    labels = df.apply(
        lambda r: heuristic_turn_label(
            r.get("description", "") if "description" in df.columns else r.get("desc", ""),
            r.get("type", ""),
            r.get("sub_type", ""),
        ),
        axis=1,
    )

    df["target"] = [x[0] for x in labels]
    df["target_source"] = [x[1] for x in labels]
    df["first_score"] = [x[2] for x in labels]
    df["second_score"] = [x[3] for x in labels]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing dataset Yu-Gi-Oh untuk klasifikasi first/second turn")
    parser.add_argument("--csv", default=None, help="Path file input CSV")
    parser.add_argument("--output", default="yugioh_preprocessed.csv", help="Path file output CSV")
    parser.add_argument("--name-col", default="name", help="Nama kolom name")
    parser.add_argument("--desc-col", default="description", help="Nama kolom description/desc")
    parser.add_argument("--target-col", default="target", help="Nama kolom target jika sudah ada")
    parser.add_argument("--drop-duplicates", action="store_true", help="Drop duplikat berdasarkan name+description")
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv)
    df = pd.read_csv(csv_path)

    if args.name_col not in df.columns:
        if "name" in df.columns:
            args.name_col = "name"
        else:
            raise ValueError(f"Kolom name tidak ditemukan. Kolom tersedia: {list(df.columns)}")

    args.desc_col = resolve_description_col(df, args.desc_col)

    df["name"] = df[args.name_col].fillna("")
    df["description"] = df[args.desc_col].fillna("")
    df["name_clean"] = df["name"].map(clean_text)
    df["description_clean"] = df["description"].map(clean_text)
    df["text_clean"] = (df["name_clean"] + " " + df["description_clean"]).str.strip()

    df["effect_tags"] = df["description"].map(extract_effect_tags)
    df["effect_tags_str"] = df["effect_tags"].map(lambda x: "|".join(x))

    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["name_clean", "description_clean"]).copy()
        print(f"[INFO] Drop duplicates: {before - len(df):,} baris dihapus")

    df = build_target(df, args.target_col)

    output_cols = [
        "name",
        "description",
        "name_clean",
        "description_clean",
        "text_clean",
        "effect_tags_str",
        "target",
        "target_source",
        "first_score",
        "second_score",
    ]

    for col in output_cols:
        if col not in df.columns:
            df[col] = None

    df[output_cols].to_csv(args.output, index=False)

    print("=" * 70)
    print("PREPROCESSING DONE")
    print("=" * 70)
    print(f"Input file   : {csv_path}")
    print(f"Output file  : {args.output}")
    print(f"Total rows   : {len(df):,}")
    print("Target distribution:")
    print(df["target"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
