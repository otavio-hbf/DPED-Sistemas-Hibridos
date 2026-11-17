import json
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


if __name__ == "__main__":
    unittest.main()
