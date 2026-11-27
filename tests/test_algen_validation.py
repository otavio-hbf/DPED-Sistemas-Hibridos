import ast
import json
from pathlib import Path
import py_compile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "algen" / "SIH-ALGEN-main" / "src"


class AlgenValidationTests(unittest.TestCase):
    def test_python_sources_compile(self):
        for filename in ("attacker_dped_sentence.py", "direct_static_decode.py"):
            py_compile.compile(str(SRC / filename), doraise=True)

    def test_validate_algen_study_notebook_structure(self):
        nb_path = ROOT / "notebooks" / "validate_algen_study.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(nb["cells"]), 8)
        all_source = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        for expected in (
            "oracle_target",
            "bert_contextual",
            "bert_pretrained",
            "--center",
            "--normalize_aligned",
            "direct_static_decode.py",
        ):
            self.assertIn(expected, all_source)

    def test_omp_decode_recovers_known_tokens(self):
        source = (SRC / "direct_static_decode.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        funcs = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("omp_decode_sentence", "_word_tokens", "token_set_metrics")
        ]
        ns = {"np": np, "STOPWORDS": frozenset({"the", "a"})}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(SRC), "exec"), ns)

        rng = np.random.default_rng(42)
        E = rng.normal(size=(50, 64)).astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        cand_ids = np.arange(100, 150, dtype=np.int64)
        true_rows = [3, 17, 41]
        x = E[true_rows].mean(axis=0)
        x /= np.linalg.norm(x)

        recovered = ns["omp_decode_sentence"](x, E, cand_ids, max_steps=10, min_gain=1e-3)
        self.assertEqual(set(recovered), {int(cand_ids[r]) for r in true_rows})


if __name__ == "__main__":
    unittest.main()
