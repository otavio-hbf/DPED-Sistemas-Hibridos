import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from main import save_dped_student_embeddings


class _Tokenizer:
    def convert_ids_to_tokens(self, token_ids):
        return [f"token-{token_id}" for token_id in token_ids]


class EmbeddingExportTests(unittest.TestCase):
    def test_saves_dped_vectors_with_aligned_token_ids_and_labels(self):
        embedding_weights = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        model = SimpleNamespace(
            bert=SimpleNamespace(
                embeddings=SimpleNamespace(
                    word_embeddings=SimpleNamespace(weight=embedding_weights)
                )
            )
        )
        public_dataset = [
            {"input_ids": torch.tensor([0, 3, 2, 3])},
            {"input_ids": torch.tensor([2, 0])},
        ]

        with tempfile.TemporaryDirectory() as output_dir:
            save_dped_student_embeddings(
                model,
                _Tokenizer(),
                public_dataset,
                output_dir,
            )

            token_ids = np.load(os.path.join(output_dir, "token_ids.npy"))
            token_embeddings = np.load(os.path.join(output_dir, "token_embeddings.npy"))
            with open(os.path.join(output_dir, "token_labels.json"), encoding="utf-8") as f:
                token_labels = json.load(f)

        np.testing.assert_array_equal(token_ids, [2, 3])
        self.assertEqual(token_labels, ["token-2", "token-3"])
        np.testing.assert_array_equal(token_embeddings, embedding_weights[[2, 3]].numpy())


if __name__ == "__main__":
    unittest.main()
