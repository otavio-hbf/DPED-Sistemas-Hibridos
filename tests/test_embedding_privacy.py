import unittest
from types import SimpleNamespace

import torch
from opacus.accountants import RDPAccountant

from models import TeacherEnsemble
from privacy import (
    add_secure_gaussian_noise,
    calculate_embedding_clip_norm,
    clip_embedding_rows,
    embedding_noise_multiplier,
    privatize_embedding_table,
    validate_embedding_privacy_parameters,
)


class EmbeddingPrivacyTests(unittest.TestCase):
    def test_clips_each_embedding_row_to_the_configured_norm(self):
        embeddings = torch.tensor([[3.0, 4.0], [0.3, 0.4]])

        clipped = clip_embedding_rows(embeddings, clip_norm=2.0)

        self.assertAlmostEqual(torch.linalg.vector_norm(clipped[0]).item(), 2.0)
        torch.testing.assert_close(clipped[1], embeddings[1])

    def test_clip_norm_uses_public_embedding_norm_quantile(self):
        embeddings = torch.tensor([[3.0, 4.0], [0.0, 2.0], [0.0, 1.0]])

        clip_norm = calculate_embedding_clip_norm(embeddings, quantile=0.5)

        self.assertAlmostEqual(clip_norm, 2.0)

    def test_delta_must_be_less_than_inverse_training_set_size(self):
        with self.assertRaisesRegex(ValueError, "less than 1/N"):
            validate_embedding_privacy_parameters(
                epsilon=2.0,
                delta=0.01,
                num_tokens=10,
                num_examples=100,
            )

    def test_noise_multiplier_increases_as_epsilon_decreases(self):
        multipliers = [
            embedding_noise_multiplier(
                epsilon=epsilon,
                delta=1e-5,
                num_tokens=8,
                num_examples=1000,
            )
            for epsilon in (1.0, 2.0, 4.0, 8.0)
        ]

        self.assertEqual(multipliers, sorted(multipliers, reverse=True))

    def test_accountant_calibration_meets_requested_epsilon(self):
        epsilon = 2.0
        delta = 1e-5
        steps = 8
        multiplier = embedding_noise_multiplier(
            epsilon=epsilon,
            delta=delta,
            num_tokens=steps,
            num_examples=1000,
        )
        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=multiplier, sample_rate=1.0)

        self.assertLessEqual(accountant.get_epsilon(delta=delta), epsilon + 0.02)

    def test_gaussian_noise_is_not_reproducible_from_torch_seed(self):
        source = torch.zeros((4, 3))
        torch.manual_seed(42)
        first = add_secure_gaussian_noise(source, standard_deviation=1.0)
        torch.manual_seed(42)
        second = add_secure_gaussian_noise(source, standard_deviation=1.0)

        self.assertFalse(torch.equal(first, second))

    def test_private_table_metadata_reports_composed_release(self):
        mean = torch.zeros((8, 3))

        private_table, metadata = privatize_embedding_table(
            clipped_teacher_mean=mean,
            num_teachers=5,
            clip_norm=1.0,
            epsilon=2.0,
            delta=1e-5,
            num_examples=1000,
        )

        self.assertEqual(tuple(private_table.shape), tuple(mean.shape))
        self.assertTrue(torch.isfinite(private_table).all())
        self.assertEqual(metadata["num_tokens"], 8)
        self.assertEqual(metadata["epsilon"], 2.0)
        self.assertEqual(metadata["privacy_scope"], "embedding table only; downstream DP-SGD excluded")

    def test_teacher_aggregation_clips_each_token_before_averaging(self):
        class Teacher(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                self.bert = SimpleNamespace(
                    embeddings=SimpleNamespace(
                        word_embeddings=torch.nn.Embedding.from_pretrained(weights)
                    )
                )

        ensemble = TeacherEnsemble.__new__(TeacherEnsemble)
        ensemble.teacher_models = [
            Teacher(torch.tensor([[3.0, 4.0], [0.3, 0.4]])),
            Teacher(torch.tensor([[0.0, 2.0], [0.6, 0.8]])),
        ]
        ensemble.device = torch.device("cpu")

        mean = ensemble.aggregate_clipped_embeddings([0, 1], clip_norm=2.0)

        torch.testing.assert_close(mean, torch.tensor([[0.6, 1.8], [0.45, 0.6]]))


if __name__ == "__main__":
    unittest.main()
