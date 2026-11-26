import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from main import save_dped_student_embeddings
from run_provenance import capture_execution, export_metadata


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "algen" / "SIH-ALGEN-main" / "src"


class Tokenizer:
    all_special_ids = [0]
    init_kwargs = {}

    def convert_ids_to_tokens(self, ids):
        return [f"token-{token_id}" for token_id in ids]


class AttackPreflightTests(unittest.TestCase):
    def test_preflight_validates_real_exports_and_detects_changed_metadata(self):
        import sys
        sys.path.insert(0, str(SOURCE))
        from embedding_provenance import (
            array_sha256, file_sha256, verify_distinct_conditions, verify_dped_research,
        )

        source = (SOURCE / "attacker_dped_sentence.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("load_static_embeddings", "preflight_conditions")]
        bert = SimpleNamespace(embeddings=SimpleNamespace(
            word_embeddings=SimpleNamespace(weight=torch.arange(4, dtype=torch.float32).reshape(2, 2) + 100)
        ))
        namespace = {
            "os": os, "np": np, "torch": torch, "hashlib": hashlib, "json": json,
            "AutoModel": SimpleNamespace(from_pretrained=lambda _: bert),
            "AutoTokenizer": SimpleNamespace(from_pretrained=lambda _: Tokenizer()),
            "array_sha256": array_sha256, "file_sha256": file_sha256,
            "verify_distinct_conditions": verify_distinct_conditions,
            "verify_dped_research": verify_dped_research, "export_metadata": export_metadata,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
        generation = capture_execution(ROOT, {
            "method": "dp-distill", "embedding_privacy_sweep": True,
            "config": {"epsilon": 8.0, "num_teachers": 5},
        })

        with tempfile.TemporaryDirectory() as directory:
            exports = {}
            for epsilon in ("8.0", "4.0", "2.0", "1.0"):
                path = Path(directory) / epsilon
                weights = torch.arange(4, dtype=torch.float32).reshape(2, 2) + float(epsilon)
                model = SimpleNamespace(bert=SimpleNamespace(embeddings=SimpleNamespace(
                    word_embeddings=SimpleNamespace(weight=weights)
                )))
                metadata = {
                    "epsilon": float(epsilon), "delta": 1e-5, "num_teachers": 5,
                    "clip_norm": 1.0, "sensitivity_per_token": 0.4,
                    "noise_multiplier": 2.0, "noise_std": 0.8,
                    "num_tokens": 2, "num_private_examples": 100,
                }
                save_dped_student_embeddings(
                    model, Tokenizer(), [0, 1], str(path), privacy_metadata=metadata,
                    config=SimpleNamespace(epsilon=float(epsilon), num_teachers=5),
                    generation_provenance=generation,
                )
                exports[epsilon] = str(path)

            conditions = namespace["preflight_conditions"](exports)
            self.assertEqual(len(conditions), 5)
            self.assertEqual(len({entry["table_array_sha256"] for entry in conditions.values()}), 5)
            metadata_path = Path(exports["4.0"]) / "metadata.txt"
            metadata_path.write_text(metadata_path.read_text(encoding="utf-8").replace("epsilon=4.0", "epsilon=8.0"),
                                     encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "epsilon label"):
                namespace["preflight_conditions"](exports)


if __name__ == "__main__":
    unittest.main()
