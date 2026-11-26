import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "analise_resultados_ataque.ipynb"


def load_notebook_functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "c004")
    tree = ast.parse("".join(cell["source"]))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"os": os, "json": json, "pd": pd, "np": np, "re": re,
                 "ALLOW_UNVERIFIED_LEGACY": False}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


class AnalysisLoadingTests(unittest.TestCase):
    def setUp(self):
        self.functions = load_notebook_functions()

    def write_run(self, root, path, *, kind=None, epsilon="8.0", n=10, selection=None):
        directory = Path(root) / path
        directory.mkdir(parents=True)
        summary = {"source": "dped", "epsilon": epsilon, "train_samples": n}
        if kind is not None:
            summary["run_kind"] = kind
        if selection is not None:
            summary["seed"] = 0
            summary["sentence_selection"] = selection
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        pd.DataFrame([{"sentence_id": 1}]).to_csv(directory / "sentence_results.csv", index=False)

    def test_excludes_smokes_and_contaminated_legacy_cell(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_run(root, "smoke", kind="research")
            self.write_run(root, "hidden_smoke_flag", kind="smoke", n=100)
            self.write_run(root, "unknown", kind="unclassified", n=100)
            self.write_run(root, "legacy_eps8p0_n10")
            self.write_run(root, "legacy_eps4p0_n10", epsilon="4.0")
            self.write_run(root, "new_eps8p0_n10", kind="research")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                runs, sentences = self.functions["load_runs"](root)

        self.assertEqual(set(runs["rel_path"]), {"new_eps8p0_n10"})
        self.assertEqual(set(sentences["rel_path"]), set(runs["rel_path"]))
        self.assertIn("Excluída execução histórica sem classificação", output.getvalue())

    def test_explicit_legacy_opt_in_never_restores_contaminated_cell(self):
        self.functions["ALLOW_UNVERIFIED_LEGACY"] = True
        with tempfile.TemporaryDirectory() as root:
            self.write_run(root, "legacy_eps8p0_n10")
            self.write_run(root, "legacy_eps4p0_n10", epsilon="4.0")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                runs, _ = self.functions["load_runs"](root)
        self.assertEqual(set(runs["rel_path"]), {"legacy_eps4p0_n10"})
        self.assertIn("execuções históricas sem classificação", output.getvalue())

    def test_rejects_unknown_run_kind(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_run(root, "run", kind="unexpected")
            with self.assertRaisesRegex(ValueError, "Tipo de execução desconhecido"):
                self.functions["load_runs"](root)

    def test_rejects_different_selection_across_conditions(self):
        first = {"test_selection_sha256": "test", "train_selection_sha256": "train"}
        for field in ("test_selection_sha256", "train_selection_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                self.write_run(root, "first", kind="research", selection=first)
                self.write_run(root, "second", kind="research",
                               selection={**first, field: "changed"})
                with self.assertRaisesRegex(ValueError, "Seleção .* divergente"):
                    self.functions["load_runs"](root)


if __name__ == "__main__":
    unittest.main()
