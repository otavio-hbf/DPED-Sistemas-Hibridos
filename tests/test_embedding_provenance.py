import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "algen" / "SIH-ALGEN-main" / "src" / "embedding_provenance.py"
)
spec = importlib.util.spec_from_file_location("embedding_provenance", MODULE_PATH)
embedding_provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embedding_provenance)


class EmbeddingProvenanceTests(unittest.TestCase):
    def test_array_hash_includes_values_shape_and_dtype(self):
        array = np.array([[1, 2], [3, 4]], dtype=np.float32)
        digest = embedding_provenance.array_sha256(array)

        self.assertEqual(digest, embedding_provenance.array_sha256(array.copy()))
        self.assertNotEqual(digest, embedding_provenance.array_sha256(array.reshape(4)))
        self.assertNotEqual(digest, embedding_provenance.array_sha256(array.astype(np.float64)))
        array[0, 0] = 5
        self.assertNotEqual(digest, embedding_provenance.array_sha256(array))

    def test_file_hash_matches_sha256_of_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token_embeddings.npy"
            np.save(path, np.array([[1, 2]], dtype=np.float32))
            self.assertEqual(
                embedding_provenance.file_sha256(path),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_preflight_rejects_duplicate_tables_and_mismatched_ids(self):
        conditions = {
            "dped:8.0": {"table_array_sha256": "table-8", "token_ids_array_sha256": "ids"},
            "dped:4.0": {"table_array_sha256": "table-4", "token_ids_array_sha256": "ids"},
            "dped:2.0": {"table_array_sha256": "table-2", "token_ids_array_sha256": "ids"},
            "dped:1.0": {"table_array_sha256": "table-1", "token_ids_array_sha256": "ids"},
            "bert": {"table_array_sha256": "bert", "token_ids_array_sha256": "ids"},
        }
        embedding_provenance.verify_distinct_conditions(conditions)

        conditions["bert"]["table_array_sha256"] = "table-8"
        with self.assertRaisesRegex(ValueError, "dped:8.0 and bert"):
            embedding_provenance.verify_distinct_conditions(conditions)

        conditions["bert"]["table_array_sha256"] = "bert"
        conditions["dped:1.0"]["table_array_sha256"] = "table-2"
        with self.assertRaisesRegex(ValueError, "dped:2.0 and dped:1.0"):
            embedding_provenance.verify_distinct_conditions(conditions)

        conditions["dped:1.0"]["table_array_sha256"] = "table-1"
        conditions["bert"]["token_ids_array_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Token IDs differ for condition bert"):
            embedding_provenance.verify_distinct_conditions(conditions)

    def test_loaded_table_and_ids_must_match_preflight(self):
        provenance = {"table_array_sha256": "table", "token_ids_array_sha256": "ids",
                      "export_provenance_sha256": "export"}
        embedding_provenance.verify_expected_provenance(provenance, "table", "ids", "export")
        for table, ids, key in (
            ("changed", "ids", "table_array_sha256"),
            ("table", "changed", "token_ids_array_sha256"),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    embedding_provenance.verify_expected_provenance(provenance, table, ids)
        with self.assertRaisesRegex(ValueError, "export_provenance_sha256"):
            embedding_provenance.verify_expected_provenance(provenance, "table", "ids", "changed")

    def test_sentence_selection_hash_detects_changed_ids_order_and_text(self):
        sentences = ["first", "second", "third"]
        selection = embedding_provenance.sentence_selection_sha256([0, 1], sentences)
        self.assertEqual(selection, embedding_provenance.sentence_selection_sha256([0, 1], sentences))
        self.assertNotEqual(selection, embedding_provenance.sentence_selection_sha256([1, 0], sentences))
        self.assertNotEqual(selection, embedding_provenance.sentence_selection_sha256([0, 2], sentences))
        self.assertNotEqual(selection, embedding_provenance.sentence_selection_sha256([0, 1], ["first", "changed", "third"]))

    def test_checkpoint_provenance_tracks_selected_checkpoint_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            checkpoint = path / "decoder.pt"
            checkpoint.write_bytes(b"checkpoint")
            config = path / "training_args_and_best_models.json"
            config.write_text(json.dumps({"best_models": [[0.1, {}, str(checkpoint)]]}), encoding="utf-8")
            provenance = embedding_provenance.decoder_checkpoint_provenance(path)
            self.assertEqual(provenance["checkpoint_sha256"], hashlib.sha256(b"checkpoint").hexdigest())
            self.assertEqual(provenance["checkpoint_path"], str(checkpoint))
            checkpoint.write_bytes(b"changed")
            self.assertNotEqual(provenance, embedding_provenance.decoder_checkpoint_provenance(path))
            config.write_text(json.dumps({"best_models": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No best decoder checkpoint"):
                embedding_provenance.decoder_checkpoint_provenance(path)

    def test_research_requires_matching_export_metadata_and_generation_artifacts(self):
        values = {
            "epsilon": "8.0", "delta": "1e-5", "num_teachers": "5", "clip_norm": "1",
            "sensitivity_per_token": "0.4", "noise_multiplier": "2", "noise_std": "0.8",
            "num_tokens": "2", "num_private_examples": "100",
            "token_embeddings.npy_sha256": "table", "token_ids.npy_sha256": "ids",
        }
        provenance = {
            "export_metadata": {"values": values},
            "table_file_sha256": "table", "token_ids_file_sha256": "ids",
            "table_shape": [2, 4],
            "generation_config": {"epsilon": 8.0, "num_teachers": 5},
            "generation_files": {
                name: "hash" for name in (
                    "generation_config.json", "generation_provenance.json",
                    "generation_code.zip", "generation_requirements.observed.txt",
                )
            },
            "generation_manifest": {
                "arguments": {"config": {"epsilon": 8.0, "num_teachers": 5}},
                "files": {"generation_code.zip": "hash", "generation_requirements.observed.txt": "hash"},
            },
        }
        embedding_provenance.verify_dped_research(provenance, "8.0")
        with self.assertRaisesRegex(ValueError, "epsilon label"):
            embedding_provenance.verify_dped_research(provenance, "4.0")
        values["token_embeddings.npy_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "hash missing or mismatched"):
            embedding_provenance.verify_dped_research(provenance, "8.0")
        values["token_embeddings.npy_sha256"] = "table"
        del provenance["generation_files"]["generation_code.zip"]
        with self.assertRaisesRegex(ValueError, "generation_code.zip"):
            embedding_provenance.verify_dped_research(provenance, "8.0")


if __name__ == "__main__":
    unittest.main()
