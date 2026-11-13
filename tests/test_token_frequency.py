import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from models import TeacherEnsemble


class _TokenDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"input_ids": torch.tensor(self.rows[index], dtype=torch.long)}


def _legacy_token_frequency(public_dataset, teacher_datasets, batch_size):
    token_counts = {}
    public_loader = DataLoader(public_dataset, batch_size=batch_size, shuffle=False)

    for batch in public_loader:
        for token_id in batch["input_ids"].unique().numpy():
            token_id = int(token_id)
            token_counts.setdefault(token_id, 0)
            for teacher_dataset in teacher_datasets:
                teacher_loader = DataLoader(teacher_dataset, batch_size=16, shuffle=False)
                for teacher_batch in teacher_loader:
                    if token_id in teacher_batch["input_ids"].unique():
                        token_counts[token_id] += 1
                        break

    return token_counts


class TokenFrequencyTests(unittest.TestCase):
    def test_counts_teacher_presence_once_across_public_batches_on_cpu(self):
        ensemble = TeacherEnsemble.__new__(TeacherEnsemble)
        ensemble.config = SimpleNamespace(batch_size=2)
        ensemble.teacher_datasets = [
            _TokenDataset([[1, 1, 8], [2, 2, 0]]),
            _TokenDataset([[3, 4, 4], [7, 0, 9]]),
            _TokenDataset([[8, 8, 8]]),
        ]
        public_dataset = _TokenDataset([[0, 1, 1], [2, 3, 0], [7, 7, 3]])
        legacy_counts = _legacy_token_frequency(
            public_dataset,
            ensemble.teacher_datasets,
            ensemble.config.batch_size,
        )

        with patch.object(torch.Tensor, "to", side_effect=AssertionError("unexpected device transfer")):
            token_counts = ensemble.get_token_frequency(public_dataset)

        self.assertEqual(
            token_counts,
            {0: 2, 1: 1, 2: 1, 3: 1, 7: 1},
        )
        self.assertEqual(
            legacy_counts,
            {0: 2, 1: 1, 2: 1, 3: 2, 7: 1},
        )

    def test_empty_public_dataset_returns_no_counts(self):
        ensemble = TeacherEnsemble.__new__(TeacherEnsemble)
        ensemble.config = SimpleNamespace(batch_size=2)
        ensemble.teacher_datasets = [_TokenDataset([[1, 2, 3]])]

        self.assertEqual(ensemble.get_token_frequency(_TokenDataset([])), {})


if __name__ == "__main__":
    unittest.main()
