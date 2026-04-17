import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC


EFFECT_TAG_PATTERNS: Dict[str, str] = {
    "search": r"\b(add|search)\b.*\b(deck|hand)\b|\badd 1\b",
    "negate": r"\bnegate\b",
    "destroy": r"\bdestroy\b",
    "banish": r"\bbanish\b",
    "draw": r"\bdraw\b",
    "bounce": r"\breturn\b.*\b(hand)\b|\bshuffle\b.*\b(deck)\b",
    "special_summon": r"\bspecial summon\b",
    "send_gy": r"\bsend\b.*\bgraveyard|\bGY\b",
    "recycle": r"\badd\b.*\bgraveyard\b|\bshuffle\b.*\bgraveyard\b",
    "lock": r"\bcannot\b|\bneither player can\b",
}


@dataclass
class ModelResult:
    model_name: str
    test_accuracy: float
    test_f1: float
    cv_f1_mean: float
    cv_f1_std: float


def clean_text(text: str) -> str:
    """Basic text cleaning for card name/effect text."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_effect_tags(text: str, patterns: Dict[str, str]) -> List[str]:
    text = "" if pd.isna(text) else str(text).lower()
    tags = [tag for tag, pattern in patterns.items() if re.search(pattern, text)]
    return tags if tags else ["other"]


def make_text_input(X: pd.DataFrame) -> List[str]:
    """Combine and clean name + description as a single text channel."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    name_series = X.iloc[:, 0].fillna("").map(clean_text)
    desc_series = X.iloc[:, 1].fillna("").map(clean_text)
    return (name_series + " " + desc_series).tolist()


def get_feature_insights(model_pipeline: Pipeline, top_n: int = 20) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    tfidf: TfidfVectorizer = model_pipeline.named_steps["tfidf"]
    clf = model_pipeline.named_steps["clf"]

    if not hasattr(clf, "coef_"):
        return [], []

    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]

    top_second_idx = np.argsort(coefs)[-top_n:][::-1]
    top_first_idx = np.argsort(coefs)[:top_n]

    top_second = [(feature_names[i], float(coefs[i])) for i in top_second_idx]
    top_first = [(feature_names[i], float(coefs[i])) for i in top_first_idx]
    return top_first, top_second


def build_pipeline(model_name: str) -> Pipeline:
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=42)
    elif model_name == "linearsvc":
        clf = LinearSVC(random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            ("combine", FunctionTransformer(make_text_input, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("clf", clf),
        ]
    )


def resolve_columns(df: pd.DataFrame, name_col: str, desc_col: str, target_col: str) -> Tuple[str, str, str]:
    if name_col not in df.columns:
        if "name" in df.columns:
            name_col = "name"
        else:
            raise ValueError(f"Kolom nama tidak ditemukan. Kandidat tersedia: {list(df.columns)}")

    if desc_col not in df.columns:
        if "description" in df.columns:
            desc_col = "description"
        elif "desc" in df.columns:
            desc_col = "desc"
        else:
            raise ValueError(f"Kolom deskripsi tidak ditemukan. Kandidat tersedia: {list(df.columns)}")

    if target_col not in df.columns:
        raise ValueError(
            "Kolom target tidak ditemukan. Tambahkan kolom label biner (0=first turn, 1=second turn), "
            f"lalu jalankan ulang. Kolom yang ada: {list(df.columns)}"
        )

    return name_col, desc_col, target_col


def run_training(
    csv_path: str,
    name_col: str,
    desc_col: str,
    target_col: str,
    test_size: float,
    cv_folds: int,
    top_n_words: int,
    save_dir: str,
) -> None:
    df = pd.read_csv(csv_path)
    name_col, desc_col, target_col = resolve_columns(df, name_col, desc_col, target_col)

    df = df[[name_col, desc_col, target_col]].dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)

    if not set(df[target_col].unique()).issubset({0, 1}):
        raise ValueError("Target harus biner dengan nilai 0 (first turn) dan 1 (second turn).")

    X = df[[name_col, desc_col]]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    model_configs = {
        "Logistic Regression": "logreg",
        "LinearSVC": "linearsvc",
    }

    results: List[ModelResult] = []
    fitted_models: Dict[str, Pipeline] = {}

    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Rows used          : {len(df):,}")
    print(f"Train size         : {len(X_train):,}")
    print(f"Test size          : {len(X_test):,}")
    print(f"CV folds           : {cv_folds}")
    print(f"Name column        : {name_col}")
    print(f"Description column : {desc_col}")
    print(f"Target column      : {target_col}")

    for display_name, key in model_configs.items():
        print("\n" + "-" * 80)
        print(f"Model: {display_name}")

        pipe = build_pipeline(key)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        print(f"CV F1 mean +- std : {cv_scores.mean():.4f} +- {cv_scores.std():.4f}")
        print(f"Test Accuracy     : {test_acc:.4f}")
        print(f"Test F1-score     : {test_f1:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred, digits=4))

        results.append(
            ModelResult(
                model_name=display_name,
                test_accuracy=test_acc,
                test_f1=test_f1,
                cv_f1_mean=float(cv_scores.mean()),
                cv_f1_std=float(cv_scores.std()),
            )
        )
        fitted_models[display_name] = pipe

    comparison = pd.DataFrame([r.__dict__ for r in results]).sort_values(by="test_f1", ascending=False)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(comparison.to_string(index=False))

    best_model_name = comparison.iloc[0]["model_name"]
    best_model = fitted_models[best_model_name]

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_obj in fitted_models.items():
        file_name = model_name.lower().replace(" ", "_") + "_pipeline.joblib"
        dump(model_obj, output_dir / file_name)

    best_path = output_dir / "best_pipeline.joblib"
    dump(best_model, best_path)

    print("\nSaved artifacts:")
    print(f"  Best pipeline      : {best_path}")
    for model_name in fitted_models:
        file_name = model_name.lower().replace(" ", "_") + "_pipeline.joblib"
        print(f"  {model_name:<18}: {output_dir / file_name}")

    print("\n" + "=" * 80)
    print(f"WORD INSIGHT FROM BEST MODEL: {best_model_name}")
    print("=" * 80)
    top_first, top_second = get_feature_insights(best_model, top_n=top_n_words)

    if top_first and top_second:
        print("Top words pushing prediction to FIRST TURN (class 0):")
        for word, coef in top_first:
            print(f"  {word:<25} {coef:.4f}")

        print("\nTop words pushing prediction to SECOND TURN (class 1):")
        for word, coef in top_second:
            print(f"  {word:<25} {coef:.4f}")
    else:
        print("Model tidak menyediakan koefisien fitur.")

    # Optional effect-tag insight (rule-based)
    df["effect_tags"] = df[desc_col].map(lambda t: extract_effect_tags(t, EFFECT_TAG_PATTERNS))
    exploded = df[[target_col, "effect_tags"]].explode("effect_tags").reset_index(drop=True)

    tag_counts = exploded["effect_tags"].value_counts().rename("count")
    tag_dist = pd.crosstab(
        exploded["effect_tags"].to_numpy(),
        exploded[target_col].to_numpy(),
        normalize="columns",
    )
    tag_dist.index.name = "effect_tags"
    tag_dist.columns.name = target_col

    print("\n" + "=" * 80)
    print("EFFECT TAG INSIGHT (OPTIONAL, RULE-BASED)")
    print("=" * 80)
    print("Top tag frequency:")
    print(tag_counts.head(20).to_string())

    print("\nTag distribution by class (column-normalized):")
    print(tag_dist.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yu-Gi-Oh first-turn vs second-turn NLP classifier")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path ke file CSV. Jika kosong, script akan mencoba auto-detect file CSV di folder saat ini.",
    )
    parser.add_argument("--name-col", default="name", help="Nama kolom nama kartu")
    parser.add_argument("--desc-col", default="desc", help="Nama kolom deskripsi efek kartu")
    parser.add_argument("--target-col", default="target", help="Nama kolom target (0/1)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Porsi test split")
    parser.add_argument("--cv-folds", type=int, default=5, help="Jumlah fold cross-validation")
    parser.add_argument("--top-n-words", type=int, default=20, help="Jumlah kata insight yang ditampilkan")
    parser.add_argument("--save-dir", default="artifacts", help="Folder output untuk menyimpan model pipeline")
    return parser.parse_args()


def resolve_csv_path(csv_arg: str | None) -> str:
    if csv_arg:
        return csv_arg

    cwd = Path.cwd()
    candidates = sorted(cwd.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        raise ValueError("Tidak ada file CSV di folder saat ini. Gunakan --csv untuk menentukan path file.")

    selected = candidates[0]
    print(f"[INFO] Argumen --csv tidak diketik, jadi file dipilih otomatis: {selected}")
    return str(selected)


if __name__ == "__main__":
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    run_training(
        csv_path=csv_path,
        name_col=args.name_col,
        desc_col=args.desc_col,
        target_col=args.target_col,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        top_n_words=args.top_n_words,
        save_dir=args.save_dir,
    )
