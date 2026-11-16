import unittest

import torch
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset, TensorDataset

from training import (
    _OpacusCompatibleDataset,
    _restore_opacus_batch,
    _set_empty_grad_samples,
)


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

    def test_empty_batch_runs_noisy_dp_step_and_updates_accountant(self):
        model = torch.nn.Linear(2, 2)
        model.train()
        data = TensorDataset(torch.randn(4, 2), torch.zeros(4, dtype=torch.long))
        loader = DataLoader(data, batch_size=2)
        optimizer = SGD(model.parameters(), lr=0.1)
        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, _ = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=1,
            target_epsilon=8.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
        )

        _set_empty_grad_samples(model)
        optimizer.step()
        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)

        self.assertGreater(epsilon, 0)
        optimizer.zero_grad()


if __name__ == "__main__":
    unittest.main()
