import json
import hashlib
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from main import _full_tokenizer_ids, save_dped_student_embeddings


class _Tokenizer:
    def get_vocab(self):
        return {f"token-{token_id}": token_id for token_id in range(5)}

    def convert_ids_to_tokens(self, token_ids):
        return [f"token-{token_id}" for token_id in token_ids]


class EmbeddingExportTests(unittest.TestCase):
    def test_token_ids_come_from_the_complete_fixed_vocabulary(self):
        token_ids = _full_tokenizer_ids(_Tokenizer(), embedding_rows=5)

        self.assertEqual(token_ids, [0, 1, 2, 3, 4])

    def test_rejects_tokenizer_vocabulary_that_does_not_match_model(self):
        with self.assertRaisesRegex(ValueError, "must exactly match"):
            _full_tokenizer_ids(_Tokenizer(), embedding_rows=4)

    def test_saves_dped_vectors_with_aligned_token_ids_and_labels(self):
        embedding_weights = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        model = SimpleNamespace(
            bert=SimpleNamespace(
                embeddings=SimpleNamespace(
                    word_embeddings=SimpleNamespace(weight=embedding_weights)
                )
            )
        )
        token_ids = [0, 1, 2, 3, 4]

        with tempfile.TemporaryDirectory() as output_dir:
            save_dped_student_embeddings(
                model,
                _Tokenizer(),
                token_ids,
                output_dir,
            )

            token_ids = np.load(os.path.join(output_dir, "token_ids.npy"))
            token_embeddings = np.load(os.path.join(output_dir, "token_embeddings.npy"))
            with open(os.path.join(output_dir, "token_labels.json"), encoding="utf-8") as f:
                token_labels = json.load(f)

        np.testing.assert_array_equal(token_ids, [0, 1, 2, 3, 4])
        self.assertEqual(token_labels, ["token-0", "token-1", "token-2", "token-3", "token-4"])
        np.testing.assert_array_equal(token_embeddings, embedding_weights.numpy())

    def test_saves_generation_config_and_export_hashes(self):
        model = SimpleNamespace(bert=SimpleNamespace(embeddings=SimpleNamespace(
            word_embeddings=SimpleNamespace(weight=torch.arange(20, dtype=torch.float32).reshape(5, 4))
        )))
        config = SimpleNamespace(seed=42, num_teachers=5, epsilon=8.0)
        with tempfile.TemporaryDirectory() as output_dir:
            save_dped_student_embeddings(
                model, _Tokenizer(), list(range(5)), output_dir,
                privacy_metadata={"noise_std": 0.25}, config=config,
            )
            with open(os.path.join(output_dir, "generation_config.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f), vars(config))
            with open(os.path.join(output_dir, "metadata.txt"), encoding="utf-8") as f:
                metadata = dict(line.rstrip("\n").split("=", 1) for line in f)
            for name in ("token_embeddings.npy", "token_ids.npy"):
                with open(os.path.join(output_dir, name), "rb") as f:
                    self.assertEqual(metadata[f"{name}_sha256"], hashlib.sha256(f.read()).hexdigest())
            self.assertEqual(metadata["noise_std"], "0.25")


if __name__ == "__main__":
    unittest.main()
