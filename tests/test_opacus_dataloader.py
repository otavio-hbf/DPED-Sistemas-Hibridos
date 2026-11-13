import unittest

import torch
from opacus.data_loader import DPDataLoader
from torch.utils.data import DataLoader, Dataset

from training import _OpacusCompatibleDataset, _restore_opacus_batch


class _MappingDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([index, 1, 2], dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "label": torch.tensor(index % 2, dtype=torch.long),
        }


class OpacusDataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.dataset = _OpacusCompatibleDataset(_MappingDataset())
        loader = DataLoader(self.dataset, batch_size=2)
        self.dp_loader = DPDataLoader.from_data_loader(loader)

    def test_empty_batch_has_valid_tensor_dtypes_and_mapping_keys(self):
        batch = _restore_opacus_batch(self.dp_loader.collate_fn([]), self.dataset.keys)

        self.assertEqual(set(batch), {"input_ids", "attention_mask", "label"})
        self.assertEqual(tuple(batch["input_ids"].shape), (0, 3))
        self.assertEqual(tuple(batch["label"].shape), (0,))
        self.assertTrue(all(value.dtype == torch.long for value in batch.values()))

    def test_nonempty_batch_restores_mapping_keys(self):
        samples = [self.dataset[0], self.dataset[1]]
        batch = _restore_opacus_batch(self.dp_loader.collate_fn(samples), self.dataset.keys)

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(tuple(batch["label"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
