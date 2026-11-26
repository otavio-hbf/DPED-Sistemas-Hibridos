import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class NotebookGridTests(unittest.TestCase):
    def test_both_notebooks_bind_preflight_and_check_paired_selection(self):
        names = ("attack_algen_dped.ipynb", "attack_algen_dped_calibracao_ruido.ipynb")
        for name in names:
            with self.subTest(notebook=name), tempfile.TemporaryDirectory() as directory:
                notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
                cells = {cell.get("metadata", {}).get("id"): "".join(cell.get("source", []))
                         for cell in notebook["cells"]}
                attacker = ModuleType("attacker_dped_sentence")
                def preflight(exports):
                    conditions = {
                        f"dped:{epsilon}": {
                            "table_array_sha256": epsilon,
                            "token_ids_array_sha256": "ids",
                            "export_provenance_sha256": "export-" + epsilon,
                        } for epsilon in exports
                    }
                    conditions["bert_pretrained"] = {
                        "table_array_sha256": "bert", "token_ids_array_sha256": "ids",
                    }
                    return conditions
                attacker.preflight_conditions = preflight
                state = {"mismatch": False, "commands": []}

                def run(command, check):
                    state["commands"].append(command)
                    output = Path(command[command.index("--output_dir") + 1])
                    output.mkdir(parents=True, exist_ok=True)
                    n = int(command[command.index("--train_samples") + 1])
                    epsilon = command[command.index("--epsilon") + 1] if "--epsilon" in command else None
                    summary = {
                        "train_samples": n, "seed": 0,
                        "sentence_selection": {
                            "test_selection_sha256": "different" if state["mismatch"] and epsilon == "4.0" else "test",
                            "train_selection_sha256": f"train-{n}",
                        },
                    }
                    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

                environment = {
                    "os": os, "sys": sys, "json": json, "subprocess": SimpleNamespace(run=run),
                    "ATTACKER": str(ROOT / "algen" / "SIH-ALGEN-main" / "src" / "attacker_dped_sentence.py"),
                    "EPSILONS": ["8.0", "4.0", "2.0", "1.0"],
                    "export_dir_for": lambda epsilon: epsilon,
                    "EXPORT_EPS8": "8.0", "ALGEN_CKPT": "checkpoint",
                    "TRAIN_SIZES": [10], "RIDGE_VALUES": [1.0],
                }
                previous = os.getcwd()
                try:
                    os.chdir(directory)
                    with mock.patch.dict(sys.modules, {"attacker_dped_sentence": attacker}), \
                            contextlib.redirect_stdout(io.StringIO()):
                        exec(cells["attack_provenance_preflight"], environment)
                        exec(cells["attack_grid"], environment)
                        self.assertEqual(len(state["commands"]), 5)
                        for command in state["commands"]:
                            self.assertIn("--expected_table_sha256", command)
                            self.assertIn("--expected_token_ids_sha256", command)
                            if "--epsilon" in command:
                                self.assertIn("--expected_export_provenance_sha256", command)
                        state["mismatch"] = True
                        with self.assertRaisesRegex(ValueError, "Seleção de teste divergente"):
                            exec(cells["attack_grid"], environment)
                finally:
                    os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
