import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load


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


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_effect_tags(text: str) -> List[str]:
    text = "" if pd.isna(text) else str(text).lower()
    tags = [tag for tag, pattern in EFFECT_TAG_PATTERNS.items() if re.search(pattern, text)]
    return tags if tags else ["other"]


def make_text_input(X: pd.DataFrame) -> List[str]:
    """Compatibility helper for old pickled pipelines using FunctionTransformer."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    name_series = X.iloc[:, 0].fillna("").map(clean_text)
    desc_series = X.iloc[:, 1].fillna("").map(clean_text)
    return (name_series + " " + desc_series).tolist()


# Backward compatibility for joblib artifacts trained when module name was __main__.
if "__main__" in sys.modules:
    setattr(sys.modules["__main__"], "make_text_input", make_text_input)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def to_label_text(pred_class: int) -> str:
    if pred_class == 0:
        return "FIRST TURN"
    if pred_class == 1:
        return "SECOND TURN"
    return f"UNKNOWN ({pred_class})"


def load_artifacts(
    pipeline_path: Optional[str],
    model_path: Optional[str],
    vectorizer_path: Optional[str],
):
    if pipeline_path:
        p = Path(pipeline_path)
        if not p.exists():
            raise FileNotFoundError(f"Pipeline tidak ditemukan: {p}")
        pipeline = load(p)
        return "pipeline", pipeline, None

    if model_path and vectorizer_path:
        mp = Path(model_path)
        vp = Path(vectorizer_path)
        if not mp.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {mp}")
        if not vp.exists():
            raise FileNotFoundError(f"Vectorizer tidak ditemukan: {vp}")
        model = load(mp)
        vectorizer = load(vp)
        return "separate", model, vectorizer

    # Auto-detect common artifact names from common run locations.
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    search_roots = [
        cwd,
        cwd / "artifacts",
        script_dir,
        script_dir / "artifacts",
    ]

    pipeline_filenames = [
        "best_pipeline.joblib",
        "pipeline.joblib",
        "model_pipeline.joblib",
        "best_pipeline.pkl",
        "pipeline.pkl",
    ]
    pipeline_candidates = [root / name for root in search_roots for name in pipeline_filenames]
    for candidate in pipeline_candidates:
        if candidate.exists():
            return "pipeline", load(candidate), None

    model_filenames = ["model.joblib", "model.pkl"]
    vectorizer_filenames = ["tfidf_vectorizer.joblib", "vectorizer.joblib", "tfidf_vectorizer.pkl"]
    model_candidates = [root / name for root in search_roots for name in model_filenames]
    vectorizer_candidates = [root / name for root in search_roots for name in vectorizer_filenames]

    found_model = next((p for p in model_candidates if p.exists()), None)
    found_vectorizer = next((p for p in vectorizer_candidates if p.exists()), None)
    if found_model and found_vectorizer:
        return "separate", load(found_model), load(found_vectorizer)

    raise ValueError(
        "Artifact model belum ditemukan.\n"
        "Gunakan salah satu opsi berikut:\n"
        "1) --pipeline-path <file_pipeline.joblib>\n"
        "2) --model-path <file_model.joblib> --vectorizer-path <file_vectorizer.joblib>\n"
        "\n"
        "Atau simpan artifact ke folder ini dengan nama umum, misalnya: best_pipeline.joblib"
    )


def _pipeline_predict_inputs(pipeline, card_name: str, card_desc: str, combined_text: str):
    """Try 2-column dataframe first (training-compatible), then fallback to plain text list."""
    two_col_input = pd.DataFrame([[card_name, card_desc]], columns=["name", "description"])
    try:
        _ = pipeline.predict(two_col_input)
        return two_col_input
    except Exception:
        return [combined_text]


def predict_with_pipeline(pipeline, card_name: str, card_desc: str, combined_text: str) -> Tuple[int, Optional[float], Optional[float], str]:
    model_input = _pipeline_predict_inputs(pipeline, card_name, card_desc, combined_text)
    pred = int(pipeline.predict(model_input)[0])

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(model_input)[0]
        p_second = float(proba[1]) if len(proba) > 1 else None
        return pred, p_second, None, "exact"

    if hasattr(pipeline, "decision_function"):
        score = float(pipeline.decision_function(model_input)[0])
        # Untuk model margin-based (mis. LinearSVC), ini bukan probabilitas terkalibrasi.
        p_second_approx = float(sigmoid(score))
        return pred, p_second_approx, score, "approx"

    return pred, None, None, "unavailable"


def predict_with_separate(model, vectorizer, combined_text: str) -> Tuple[int, Optional[float], Optional[float], str]:
    X = vectorizer.transform([combined_text])
    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        p_second = float(proba[1]) if len(proba) > 1 else None
        return pred, p_second, None, "exact"

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        p_second_approx = float(sigmoid(score))
        return pred, p_second_approx, score, "approx"

    return pred, None, None, "unavailable"


def run_single_inference(mode, model_or_pipeline, vectorizer, card_name: str, card_desc: str, show_tags: bool) -> None:
    combined_text = f"{clean_text(card_name)} {clean_text(card_desc)}".strip()

    if mode == "pipeline":
        pred, p_second, score, prob_status = predict_with_pipeline(model_or_pipeline, card_name, card_desc, combined_text)
    else:
        pred, p_second, score, prob_status = predict_with_separate(model_or_pipeline, vectorizer, combined_text)

    label_text = to_label_text(pred)

    print("\n" + "=" * 72)
    print("HASIL INFERENCE KARTU YU-GI-OH")
    print("=" * 72)
    print(f"Nama Kartu      : {card_name}")
    print(f"Prediksi Kelas  : {pred} ({label_text})")

    if p_second is not None:
        print(f"Prob SECOND TURN: {p_second:.4f}")
        print(f"Prob FIRST TURN : {1.0 - p_second:.4f}")
        if prob_status == "approx":
            print("Catatan         : Probabilitas di atas adalah APPROX (dari decision score), bukan calibrated probability.")
    else:
        print("Probabilitas    : Tidak tersedia untuk model ini.")

    if score is not None:
        print(f"Decision score  : {score:.4f}")

    if show_tags:
        tags = extract_effect_tags(card_desc)
        print(f"Effect tags     : {', '.join(tags)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference first-turn vs second-turn kartu Yu-Gi-Oh")

    parser.add_argument("--pipeline-path", default=None, help="Path ke file pipeline joblib/pkl")
    parser.add_argument("--model-path", default=None, help="Path ke file model joblib/pkl")
    parser.add_argument("--vectorizer-path", default=None, help="Path ke file TF-IDF vectorizer joblib/pkl")

    parser.add_argument("--name", default=None, help="Nama kartu (opsional, jika kosong akan diminta via input)")
    parser.add_argument("--desc", default=None, help="Deskripsi efek kartu (opsional, jika kosong akan diminta via input)")
    parser.add_argument("--no-tags", action="store_true", help="Matikan tampilan tag efek")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mode, model_or_pipeline, vectorizer = load_artifacts(
        pipeline_path=args.pipeline_path,
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
    )

    card_name = args.name if args.name is not None else input("Masukkan nama kartu: ").strip()
    card_desc = args.desc if args.desc is not None else input("Masukkan deskripsi efek kartu: ").strip()

    run_single_inference(
        mode=mode,
        model_or_pipeline=model_or_pipeline,
        vectorizer=vectorizer,
        card_name=card_name,
        card_desc=card_desc,
        show_tags=not args.no_tags,
    )


if __name__ == "__main__":
    main()
