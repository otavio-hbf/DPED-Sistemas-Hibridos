"""
Direct token-set decoding from static word-embedding sentence averages (Phase 2 positive control).

Tests whether tokens can be recovered directly from the L2-normalized mean of static
word_embeddings without ALGEN (X->Y alignment + frozen flan-t5 decoder), using
Orthogonal Matching Pursuit (OMP) over the vocabulary table.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import datasets
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))


STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "for", "from", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not",
    "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "s", "same", "she", "should", "so",
    "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves",
})


def omp_decode_sentence(x_vec, E_norm, candidate_ids, max_steps=20, min_gain=1e-3):
    """Recover a set of token IDs from a normalized mean vector via Orthogonal Matching Pursuit."""
    residual = x_vec.copy()
    selected_rows = []
    selected_set = set()
    prev_err = float(np.linalg.norm(residual))

    for _ in range(max_steps):
        scores = E_norm @ residual
        for r in selected_set:
            scores[r] = -np.inf
        best_row = int(np.argmax(scores))
        if scores[best_row] <= 0:
            break
        selected_rows.append(best_row)
        selected_set.add(best_row)

        basis = E_norm[selected_rows].T  # (d, k)
        coeffs, _, _, _ = np.linalg.lstsq(basis, x_vec, rcond=None)
        approx = basis @ coeffs
        new_residual = x_vec - approx
        err = float(np.linalg.norm(new_residual))
        if prev_err - err < min_gain:
            selected_rows.pop()
            break
        residual = new_residual
        prev_err = err
        if err < 1e-6:
            break

    return [int(candidate_ids[r]) for r in selected_rows]


def _word_tokens(text):
    return [w for w in text.lower().split() if any(c.isalnum() for c in w)]


def token_set_metrics(pred_ids, ref_ids, tokenizer):
    pred_set, ref_set = set(pred_ids), set(ref_ids)
    inter = len(pred_set & ref_set)
    prec = inter / len(pred_set) if pred_set else 0.0
    rec = inter / len(ref_set) if ref_set else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    pred_words = set(_word_tokens(tokenizer.decode(pred_ids, skip_special_tokens=True)))
    ref_words = set(_word_tokens(tokenizer.decode(ref_ids, skip_special_tokens=True)))
    ref_content = {w for w in ref_words if w not in STOPWORDS}
    ref_stop = {w for w in ref_words if w in STOPWORDS}
    content_rec = len(pred_words & ref_content) / len(ref_content) if ref_content else 0.0
    stop_rec = len(pred_words & ref_stop) / len(ref_stop) if ref_stop else 0.0
    return {
        "token_precision": prec,
        "token_recall": rec,
        "token_f1": f1,
        "content_word_recall": content_rec,
        "stopword_recall": stop_rec,
    }


def main():
    parser = argparse.ArgumentParser(description="Direct OMP token-set decoding on static embeddings.")
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--export_dir", default=None, help="Optional DPED export dir; uses clean BERT if omitted.")
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--min_gain", type=float, default=1e-3)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.bert_model)
    special_ids = set(tok.all_special_ids)

    if args.export_dir:
        token_ids = np.load(os.path.join(args.export_dir, "token_ids.npy"))
        table = np.load(os.path.join(args.export_dir, "token_embeddings.npy"))
    else:
        token_ids = np.arange(tok.vocab_size, dtype=np.int64)
        bert = AutoModel.from_pretrained(args.bert_model)
        table = bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
        del bert

    keep_mask = np.array([int(t) not in special_ids for t in token_ids], dtype=bool)
    candidate_ids = token_ids[keep_mask]
    E_cand = table[keep_mask].astype(np.float32)
    norms = np.linalg.norm(E_cand, axis=1, keepdims=True)
    E_norm = E_cand / np.clip(norms, 1e-12, None)
    row_of = {int(t): i for i, t in enumerate(candidate_ids)}

    sst2 = datasets.load_dataset("glue", "sst2")
    test_sentences = sst2["validation"]["sentence"]
    test_token_lists = tok(test_sentences, add_special_tokens=False)["input_ids"]

    eligible = []
    for idx, ids in enumerate(test_token_lists):
        kept_ids = [t for t in ids if t in row_of]
        if kept_ids:
            eligible.append((idx, kept_ids))
        if len(eligible) >= args.test_samples:
            break

    rows = []
    for sent_id, ref_ids in eligible:
        vec = E_cand[[row_of[t] for t in ref_ids]].mean(axis=0)
        vec_norm = vec / max(float(np.linalg.norm(vec)), 1e-12)
        pred_ids = omp_decode_sentence(
            vec_norm, E_norm, candidate_ids, max_steps=args.max_steps, min_gain=args.min_gain
        )
        m = token_set_metrics(pred_ids, ref_ids, tok)
        rows.append({
            "sentence_id": sent_id,
            "ref_len": len(set(ref_ids)),
            "pred_len": len(pred_ids),
            "reference": test_sentences[sent_id],
            "predicted_tokens": tok.decode(pred_ids, skip_special_tokens=True),
            **m,
        })

    df = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "direct_decode_results.csv"), index=False)
    summary = {
        "source": "dped" if args.export_dir else "bert_pretrained",
        "test_samples": len(df),
        "max_steps": args.max_steps,
        "min_gain": args.min_gain,
        "mean_token_precision": float(df["token_precision"].mean()),
        "mean_token_recall": float(df["token_recall"].mean()),
        "mean_token_f1": float(df["token_f1"].mean()),
        "mean_content_word_recall": float(df["content_word_recall"].mean()),
        "mean_stopword_recall": float(df["stopword_recall"].mean()),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
