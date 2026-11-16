"""
Sentence-level ALGEN alignment attack on DPED static token embeddings.

X (source) of a sentence: mean of the DPED token rows of its content tokens whose
token_freq > 0 in the export, L2-normalized. Tokens outside the export or with
token_freq == 0 are dropped. Sentences without any such token are skipped.
Y (target): flan-t5 encoder mean-pool of the sentence, L2-normalized.
The decoder is the frozen pretrained ALGEN checkpoint; only the ridge map X->Y is fitted.

Baseline: --source bert_pretrained uses the bert-base-uncased word-embedding rows
for the same token_ids and the same token_freq mask.
"""
import argparse
import json
import os

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from attacker_gt import DecoderInference
from eval_metrics import get_rouge_scores
from inversion_utils import set_seed, add_punctuation_token_ids, get_Y_embeddings_from_tokens


def load_static_embeddings(export_dir, source, bert_model, special_ids):
    token_ids = np.load(os.path.join(export_dir, "token_ids.npy"))
    freq_path = os.path.join(export_dir, "token_freq.npy")
    if not os.path.exists(freq_path):
        raise FileNotFoundError(f"{freq_path} not found. Re-export DPED with token_freq (see exportacao-embeddings-dped.md).")
    token_freq = np.load(freq_path)
    if len(token_freq) != len(token_ids):
        raise ValueError("token_freq.npy and token_ids.npy have different lengths.")

    if source == "dped":
        table = np.load(os.path.join(export_dir, "token_embeddings.npy"))
    elif source == "bert_pretrained":
        bert = AutoModel.from_pretrained(bert_model)
        table = bert.embeddings.word_embeddings.weight.detach()[torch.as_tensor(token_ids, dtype=torch.long)].cpu().numpy()
        del bert
    else:
        raise ValueError(f"Unknown source: {source}")

    if table.shape[0] != len(token_ids):
        raise ValueError("Embedding rows and token_ids have different lengths.")

    row_of = {int(t): i for i, t in enumerate(token_ids)}
    allowed = {int(t) for t, f in zip(token_ids, token_freq) if f > 0 and int(t) not in special_ids}
    return table, row_of, allowed


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


class DPEDSentenceDecoder(DecoderInference):
    """
    Reuses the ALGEN decoder, ridge mapping and test(). Only the construction of X/Y is replaced.
    source_model_name="random" skips loading a source encoder in the parent __init__;
    initialize_embeddings is overridden, so no random vectors are used.
    """

    def __init__(self, checkpoint_path, X_train, train_texts, X_test, test_texts, reg_lambda):
        self.X_train_np = X_train
        self.train_texts = train_texts
        self.X_test_np = X_test
        self.test_texts = test_texts
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

        X_train = self.X_train_np.to(self.device)
        X_test = self.X_test_np.to(self.device)
        _, T = self.mapping_X_to_Y_pooled(X_train, Y_train, self.reg_lambda)
        self.source_hidden_dim, self.target_hidden_dim = T.shape

        X_test_aligned = X_test @ T
        self.per_sentence_cosine = F.cosine_similarity(X_test_aligned, Y_test, dim=1).detach().cpu().numpy()
        self.align_metrics = {"X_Y_test_COS": float(self.per_sentence_cosine.mean()),
                              "reg_lambda": self.reg_lambda}

        self.test_data = {
            "hidden_states": X_test_aligned,
            "attention_mask": Y_test_tokens["attention_mask"],
            "texts": tokenizer.batch_decode(Y_test_tokens["input_ids"], skip_special_tokens=True),
        }


def main():
    parser = argparse.ArgumentParser(description="Sentence-level ALGEN attack on DPED embeddings.")
    parser.add_argument("--checkpoint_path", required=True, help="ALGEN decoder checkpoint directory.")
    parser.add_argument("--export_dir", required=True, help="DPED export dir with token_ids, token_embeddings, token_freq.")
    parser.add_argument("--source", choices=["dped", "bert_pretrained"], default="dped")
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--epsilon", default=None, help="Label only, stored in the output.")
    parser.add_argument("--train_samples", type=int, required=True)
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--ridge", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    bert_tok = AutoTokenizer.from_pretrained(args.bert_model)
    special_ids = set(bert_tok.all_special_ids)

    table, row_of, allowed = load_static_embeddings(args.export_dir, args.source, args.bert_model, special_ids)
    print(f"source={args.source} rows={table.shape[0]} allowed_content_tokens={len(allowed)}")

    sst2 = datasets.load_dataset("glue", "sst2")
    train_sentences = sst2["train"]["sentence"]
    test_sentences = sst2["validation"]["sentence"]

    X_train_all, kept_train = pool_sentences(
        bert_tok(train_sentences, add_special_tokens=False)["input_ids"], table, row_of, allowed)
    X_test_all, kept_test = pool_sentences(
        bert_tok(test_sentences, add_special_tokens=False)["input_ids"], table, row_of, allowed)

    if args.train_samples > len(kept_train):
        raise ValueError(f"train_samples={args.train_samples} exceeds eligible train sentences ({len(kept_train)}).")
    if args.test_samples > len(kept_test):
        raise ValueError(f"test_samples={args.test_samples} exceeds eligible test sentences ({len(kept_test)}).")

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(kept_train), size=args.train_samples, replace=False)
    train_idx = [kept_train[i] for i in chosen]
    X_train = X_train_all[torch.as_tensor(chosen)]
    test_idx = kept_test[:args.test_samples]
    X_test = X_test_all[:args.test_samples]

    print(f"eligible train {len(kept_train)}/{len(train_sentences)}, test {len(kept_test)}/{len(test_sentences)}")

    decoder = DPEDSentenceDecoder(
        args.checkpoint_path,
        X_train, [train_sentences[i] for i in train_idx],
        X_test, [test_sentences[i] for i in test_idx],
        args.ridge,
    )
    gen_results, preds, refs = decoder.test()

    rows = []
    for k, (pred, ref, cos) in enumerate(zip(preds, refs, decoder.per_sentence_cosine)):
        rouge = get_rouge_scores([pred], [ref])
        rows.append({
            "epsilon": args.epsilon, "source": args.source, "train_samples": args.train_samples,
            "ridge": args.ridge, "seed": args.seed, "sentence_id": int(test_idx[k]),
            "prediction": pred, "reference": ref, "cosine": float(cos),
            "rouge1": rouge["rouge1_f"][0], "rougeL": rouge["rougeL_f"][0],
        })

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "sentence_results.csv"), index=False)
    summary = {
        "epsilon": args.epsilon, "source": args.source, "train_samples": args.train_samples,
        "test_samples": args.test_samples, "ridge": args.ridge, "seed": args.seed,
        "eligible_train": len(kept_train), "eligible_test": len(kept_test),
        "source_dim": decoder.source_hidden_dim, "target_dim": decoder.target_hidden_dim,
        "align_metrics": decoder.align_metrics, "test_results": gen_results,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
