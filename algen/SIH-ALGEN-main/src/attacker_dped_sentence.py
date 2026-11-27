"""
Sentence-level ALGEN alignment attack on DPED static token embeddings.

X (source) of a sentence: mean of the DPED token rows of its content tokens in
the fixed tokenizer vocabulary, excluding special tokens, L2-normalized.
Sentences without any such token are skipped.
Y (target): flan-t5 encoder mean-pool of the sentence, L2-normalized.
The decoder is the frozen pretrained ALGEN checkpoint; only the ridge map X->Y is fitted.

Baseline: --source bert_pretrained uses the bert-base-uncased word-embedding rows
for the same full-vocabulary token_ids and special-token mask.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))
from run_provenance import capture_execution, export_metadata, write_execution

from attacker_gt import DecoderInference
from embedding_provenance import (
    array_sha256, decoder_checkpoint_provenance, file_sha256, sentence_selection_sha256,
    verify_distinct_conditions, verify_dped_research, verify_expected_provenance,
)
from eval_metrics import get_rouge_scores
from inversion_utils import set_seed, add_punctuation_token_ids, get_Y_embeddings_from_tokens


def load_static_embeddings(export_dir, source, bert_model, special_ids, vocab_size=None):
    if export_dir is not None:
        token_ids_path = os.path.join(export_dir, "token_ids.npy")
        token_ids = np.load(token_ids_path)
    elif source == "bert_pretrained":
        if vocab_size is None:
            raise ValueError("vocab_size is required when export_dir is omitted for bert_pretrained.")
        token_ids_path = None
        token_ids = np.arange(vocab_size, dtype=np.int64)
    else:
        raise ValueError(f"--export_dir is required for source={source}.")

    if source == "dped":
        table_path = os.path.join(export_dir, "token_embeddings.npy")
        table = np.load(table_path)
    elif source == "bert_pretrained":
        bert = AutoModel.from_pretrained(bert_model)
        table = bert.embeddings.word_embeddings.weight.detach()[torch.as_tensor(token_ids, dtype=torch.long)].cpu().numpy()
        del bert
    else:
        raise ValueError(f"Unknown source: {source}")

    if token_ids.ndim != 1 or len(np.unique(token_ids)) != len(token_ids):
        raise ValueError("Token IDs must be a one-dimensional array of unique IDs.")
    if table.ndim != 2 or table.shape[0] != len(token_ids):
        raise ValueError("Embedding rows and token_ids have different lengths.")

    row_of = {int(t): i for i, t in enumerate(token_ids)}
    allowed = {int(token_id) for token_id in token_ids if int(token_id) not in special_ids}
    provenance = {
        "table_array_sha256": array_sha256(table),
        "table_shape": list(table.shape),
        "table_dtype": table.dtype.str,
        "token_ids_array_sha256": array_sha256(token_ids),
        "token_ids_file_sha256": file_sha256(token_ids_path) if token_ids_path is not None else None,
        "export_dir": os.path.abspath(export_dir) if export_dir is not None else None,
    }
    if source == "dped":
        provenance["table_file_sha256"] = file_sha256(table_path)
        metadata_path = os.path.join(export_dir, "metadata.txt")
        provenance["export_metadata"] = export_metadata(metadata_path) if os.path.isfile(metadata_path) else None
        provenance["generation_files"] = {}
        for name in (
            "generation_config.json", "generation_provenance.json",
            "generation_code.zip", "generation_requirements.observed.txt",
        ):
            path = os.path.join(export_dir, name)
            if os.path.isfile(path):
                provenance["generation_files"][name] = file_sha256(path)
        config_path = os.path.join(export_dir, "generation_config.json")
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as f:
                provenance["generation_config"] = json.load(f)
        generation_path = os.path.join(export_dir, "generation_provenance.json")
        if os.path.isfile(generation_path):
            with open(generation_path, encoding="utf-8") as f:
                record = json.load(f)
            provenance["generation_manifest"] = {
                "arguments": record["arguments"], "files": record["files"],
                "revision": record["code"]["revision"],
                "code_tree_sha256": record["code"]["code_tree_sha256"],
            }
        export_record = {
            "metadata": provenance["export_metadata"],
            "generation_files": provenance["generation_files"],
        }
        provenance["export_provenance_sha256"] = hashlib.sha256(
            json.dumps(export_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    else:
        provenance["bert_model"] = bert_model
    return table, row_of, allowed, provenance


def preflight_conditions(exports, bert_model="bert-base-uncased"):
    if set(exports) != {"8.0", "4.0", "2.0", "1.0"}:
        raise ValueError("Preflight requires exactly epsilon 8.0, 4.0, 2.0 and 1.0.")
    special_ids = set(AutoTokenizer.from_pretrained(bert_model).all_special_ids)
    conditions = {}
    for epsilon, export_dir in exports.items():
        table, row_of, allowed, provenance = load_static_embeddings(
            export_dir, "dped", bert_model, special_ids
        )
        verify_dped_research(provenance, epsilon)
        conditions[f"dped:{epsilon}"] = provenance
        del table, row_of, allowed
    table, row_of, allowed, provenance = load_static_embeddings(
        exports["8.0"], "bert_pretrained", bert_model, special_ids
    )
    conditions["bert_pretrained"] = provenance
    del table, row_of, allowed
    verify_distinct_conditions(conditions)
    return conditions


def pool_sentences(token_lists, table, row_of, allowed):
    """Return normalized mean vectors and the indices of the sentences that kept at least one token."""
    vectors, kept = [], []
    for idx, ids in enumerate(token_lists):
        rows = [row_of[t] for t in ids if t in allowed]
        if not rows:
            continue
        vectors.append(table[rows].mean(axis=0))
        kept.append(idx)
    X = torch.nn.functional.normalize(torch.tensor(np.stack(vectors), dtype=torch.float32), p=2, dim=1)
    return X, kept


def pool_contextual_sentences(sentences, indices, bert_model, bert_tok, max_length=32, batch_size=128):
    """Mean-pool contextual last_hidden_state from BERT for the selected sentence indices (ALGEN baseline)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AutoModel.from_pretrained(bert_model).to(device)
    encoder.eval()
    selected = [sentences[i] for i in indices]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            enc = bert_tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            hidden = encoder(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            chunks.append(F.normalize(pooled, p=2, dim=1).cpu())
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


class DPEDSentenceDecoder(DecoderInference):
    """
    Reuses the ALGEN decoder, ridge mapping and test(). Only the construction of X/Y is replaced.
    source_model_name="random" skips loading a source encoder in the parent __init__;
    initialize_embeddings is overridden, so no random vectors are used.
    """

    def __init__(self, checkpoint_path, X_train, train_texts, X_test, test_texts, reg_lambda,
                 center=False, normalize_aligned=False, oracle_mode=False):
        self.X_train_np = X_train
        self.train_texts = train_texts
        self.X_test_np = X_test
        self.test_texts = test_texts
        self.center = center
        self.normalize_aligned = normalize_aligned
        self.oracle_mode = oracle_mode
        super().__init__(checkpoint_path, "random",
                         len(train_texts), len(test_texts), "sst2",
                         regularization=True, reg_lambda=reg_lambda)

    def initialize_embeddings(self):
        tokenizer, encoder = self.target_tokenizer, self.target_encoder
        max_length = self.trainer.max_length

        Y_train_tokens = add_punctuation_token_ids(self.train_texts, tokenizer, max_length, self.device)
        Y_test_tokens = add_punctuation_token_ids(self.test_texts, tokenizer, max_length, self.device)
        Y_train = get_Y_embeddings_from_tokens(Y_train_tokens, encoder, normalization=True)
        Y_test = get_Y_embeddings_from_tokens(Y_test_tokens, encoder, normalization=True)
        self.Y_train_ref = Y_train
        self.Y_test_ref = Y_test

        if self.oracle_mode:
            X_test_aligned = Y_test
            self.source_hidden_dim, self.target_hidden_dim = Y_test.shape[1], Y_test.shape[1]
        else:
            X_train = self.X_train_np.to(self.device)
            X_test = self.X_test_np.to(self.device)
            if self.center:
                x_mean = X_train.mean(dim=0, keepdim=True)
                y_mean = Y_train.mean(dim=0, keepdim=True)
                _, T = self.mapping_X_to_Y_pooled(X_train - x_mean, Y_train - y_mean, self.reg_lambda)
                X_test_aligned = (X_test - x_mean) @ T + y_mean
            else:
                _, T = self.mapping_X_to_Y_pooled(X_train, Y_train, self.reg_lambda)
                X_test_aligned = X_test @ T
            self.source_hidden_dim, self.target_hidden_dim = T.shape

        raw_norms = X_test_aligned.norm(p=2, dim=1)
        if self.normalize_aligned and not self.oracle_mode:
            X_test_aligned = F.normalize(X_test_aligned, p=2, dim=1)

        cos_matrix = Y_test @ Y_test.T
        n_test = Y_test.shape[0]
        anisotropy = float((cos_matrix.sum() - torch.trace(cos_matrix)) / max(n_test * (n_test - 1), 1))

        self.per_sentence_cosine = F.cosine_similarity(X_test_aligned, Y_test, dim=1).detach().cpu().numpy()
        self.align_metrics = {
            "X_Y_test_COS": float(self.per_sentence_cosine.mean()),
            "reg_lambda": self.reg_lambda,
            "center": self.center,
            "normalize_aligned": self.normalize_aligned,
            "oracle_mode": self.oracle_mode,
            "aligned_norm_mean": float(raw_norms.mean()),
            "aligned_norm_std": float(raw_norms.std(unbiased=False)),
            "anisotropy_y_test_mean_cosine": anisotropy,
        }

        self.test_data = {
            "hidden_states": X_test_aligned,
            "attention_mask": Y_test_tokens["attention_mask"],
            "texts": tokenizer.batch_decode(Y_test_tokens["input_ids"], skip_special_tokens=True),
        }

    def evaluate_constant_baseline(self):
        self.trainer.model.eval()
        y_const = F.normalize(self.Y_train_ref.mean(dim=0, keepdim=True), p=2, dim=1)
        y_const_batch = y_const.expand(len(self.test_data["texts"]), -1).to(self.device)
        with torch.no_grad():
            generated_ids = self.trainer.model.generate({
                "hidden_states": y_const_batch,
                "attention_mask": self.test_data["attention_mask"].to(self.device),
            })
            const_preds = [t.strip() for t in self.trainer.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]
        const_eval = self.trainer.eval_texts(const_preds, self.test_data["texts"])
        const_cos = float(F.cosine_similarity(y_const_batch, self.Y_test_ref, dim=1).mean())
        return {
            "sample_prediction": const_preds[0] if const_preds else "",
            "cosine": const_cos,
            "test_results": const_eval,
        }


def main():
    parser = argparse.ArgumentParser(description="Sentence-level ALGEN attack on DPED embeddings.")
    parser.add_argument("--checkpoint_path", required=True, help="ALGEN decoder checkpoint directory.")
    parser.add_argument("--export_dir", default=None, help="DPED export dir (required for --source dped).")
    parser.add_argument("--source", choices=["dped", "bert_pretrained", "bert_contextual", "oracle_target"], default="dped")
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--epsilon", default=None, help="Label only, stored in the output.")
    parser.add_argument("--train_samples", type=int, required=True)
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--ridge", type=float, required=True)
    parser.add_argument("--center", action="store_true", help="Center X and Y on train mean before fitting ridge.")
    parser.add_argument("--normalize_aligned", action="store_true", help="L2-normalize X_test_aligned before decoding.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_kind", choices=["research", "smoke"], default=None,
                        help="Classify the run for research analysis; omitted runs are unclassified.")
    parser.add_argument("--expected_table_sha256", help="Expected loaded-table hash from preflight.")
    parser.add_argument("--expected_token_ids_sha256", help="Expected loaded-token-IDs hash from preflight.")
    parser.add_argument("--expected_export_provenance_sha256", help="Expected DPED metadata and generation hash.")
    args = parser.parse_args()

    if args.train_samples <= 0 or args.test_samples <= 0:
        parser.error("--train_samples and --test_samples must be positive.")
    if args.source == "dped" and not args.export_dir:
        parser.error("--export_dir is required when --source dped.")

    run_provenance = capture_execution(REPO_ROOT, args)
    set_seed(args.seed)
    bert_tok = AutoTokenizer.from_pretrained(args.bert_model)
    special_ids = set(bert_tok.all_special_ids)

    sst2 = datasets.load_dataset("glue", "sst2")
    train_sentences = sst2["train"]["sentence"]
    test_sentences = sst2["validation"]["sentence"]
    train_token_lists = bert_tok(train_sentences, add_special_tokens=False)["input_ids"]
    test_token_lists = bert_tok(test_sentences, add_special_tokens=False)["input_ids"]

    if args.source in ("dped", "bert_pretrained"):
        table, row_of, allowed, provenance = load_static_embeddings(
            args.export_dir, args.source, args.bert_model, special_ids, vocab_size=bert_tok.vocab_size
        )
        verify_expected_provenance(
            provenance, args.expected_table_sha256, args.expected_token_ids_sha256,
            args.expected_export_provenance_sha256,
        )
        if args.source == "dped" and args.run_kind == "research":
            verify_dped_research(provenance, args.epsilon)
        print(f"source={args.source} rows={table.shape[0]} allowed_content_tokens={len(allowed)}")
        print(f"table_array_sha256={provenance['table_array_sha256']}")

        X_train_all, kept_train = pool_sentences(train_token_lists, table, row_of, allowed)
        X_test_all, kept_test = pool_sentences(test_token_lists, table, row_of, allowed)
    else:
        allowed = {i for i in range(bert_tok.vocab_size) if i not in special_ids}
        kept_train = [idx for idx, ids in enumerate(train_token_lists) if any(t in allowed for t in ids)]
        kept_test = [idx for idx, ids in enumerate(test_token_lists) if any(t in allowed for t in ids)]
        provenance = {"source": args.source, "bert_model": args.bert_model}
        print(f"source={args.source} allowed_content_tokens={len(allowed)}")

    if args.train_samples > len(kept_train):
        raise ValueError(f"train_samples={args.train_samples} exceeds eligible train sentences ({len(kept_train)}).")
    if args.test_samples > len(kept_test):
        raise ValueError(f"test_samples={args.test_samples} exceeds eligible test sentences ({len(kept_test)}).")

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(kept_train), size=args.train_samples, replace=False)
    train_idx = [kept_train[i] for i in chosen]
    test_idx = kept_test[:args.test_samples]

    if args.source in ("dped", "bert_pretrained"):
        X_train = X_train_all[torch.as_tensor(chosen)]
        X_test = X_test_all[:args.test_samples]
    elif args.source == "bert_contextual":
        X_train = pool_contextual_sentences(train_sentences, train_idx, args.bert_model, bert_tok)
        X_test = pool_contextual_sentences(test_sentences, test_idx, args.bert_model, bert_tok)
    else:
        X_train = torch.empty((len(train_idx), 1), dtype=torch.float32)
        X_test = torch.empty((len(test_idx), 1), dtype=torch.float32)
    selection = {
        "train_split": "glue/sst2:train",
        "train_procedure": "numpy.default_rng(seed).choice eligible rows without replacement",
        "train_sentence_ids": [int(i) for i in train_idx],
        "train_selection_sha256": sentence_selection_sha256(train_idx, train_sentences),
        "test_split": "glue/sst2:validation",
        "test_procedure": "first eligible rows in dataset order",
        "test_sentence_ids": [int(i) for i in test_idx],
        "test_selection_sha256": sentence_selection_sha256(test_idx, test_sentences),
    }

    print(f"eligible train {len(kept_train)}/{len(train_sentences)}, test {len(kept_test)}/{len(test_sentences)}")

    checkpoint_provenance = decoder_checkpoint_provenance(args.checkpoint_path)
    decoder = DPEDSentenceDecoder(
        args.checkpoint_path,
        X_train, [train_sentences[i] for i in train_idx],
        X_test, [test_sentences[i] for i in test_idx],
        args.ridge,
        center=args.center,
        normalize_aligned=args.normalize_aligned,
        oracle_mode=(args.source == "oracle_target"),
    )
    gen_results, preds, refs = decoder.test()
    constant_baseline = decoder.evaluate_constant_baseline()
    if not (len(preds) == len(refs) == len(decoder.per_sentence_cosine) == len(test_idx)):
        raise RuntimeError("Decoder output count does not match selected test sentences.")
    if (os.path.abspath(decoder.best_models[0][-1]) != checkpoint_provenance["checkpoint_path"]
            or decoder_checkpoint_provenance(args.checkpoint_path) != checkpoint_provenance):
        raise RuntimeError("Decoder checkpoint or configuration changed during the attack.")

    rows = []
    for k, (pred, ref, cos) in enumerate(zip(preds, refs, decoder.per_sentence_cosine)):
        rouge = get_rouge_scores([pred], [ref])
        rows.append({
            "epsilon": args.epsilon, "source": args.source, "train_samples": args.train_samples,
            "ridge": args.ridge, "seed": args.seed, "run_kind": args.run_kind or "unclassified",
            "sentence_id": int(test_idx[k]),
            "prediction": pred, "reference": ref, "cosine": float(cos),
            "rouge1": rouge["rouge1_f"][0], "rougeL": rouge["rougeL_f"][0],
        })

    os.makedirs(args.output_dir, exist_ok=True)
    write_execution(REPO_ROOT, run_provenance, args.output_dir, "attack")
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "sentence_results.csv"), index=False)
    summary = {
        "epsilon": args.epsilon, "source": args.source, "train_samples": args.train_samples,
        "test_samples": args.test_samples, "ridge": args.ridge, "seed": args.seed,
        "run_kind": args.run_kind or "unclassified",
        "eligible_train": len(kept_train), "eligible_test": len(kept_test),
        "source_dim": decoder.source_hidden_dim, "target_dim": decoder.target_hidden_dim,
        "embedding_provenance": provenance,
        "decoder_provenance": checkpoint_provenance,
        "sentence_selection": selection,
        "external_resources": {
            "dataset": "glue/sst2",
            "dataset_train_fingerprint": sst2["train"]._fingerprint,
            "dataset_validation_fingerprint": sst2["validation"]._fingerprint,
            "bert_tokenizer_revision": bert_tok.init_kwargs.get("_commit_hash"),
            "target_model": decoder.target_model_name,
            "target_model_revision": getattr(decoder.trainer.model.encoder_decoder.config, "_commit_hash", None),
            "target_tokenizer_revision": decoder.target_tokenizer.init_kwargs.get("_commit_hash"),
        },
        "align_metrics": decoder.align_metrics,
        "distinct_predictions": len(set(preds)),
        "constant_baseline": constant_baseline,
        "test_results": gen_results,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
