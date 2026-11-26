import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SOURCE_DIR = Path(__file__).resolve().parents[1] / "algen" / "SIH-ALGEN-main" / "src"
sys.path.insert(0, str(SOURCE_DIR))
spec = importlib.util.spec_from_file_location("audit_historical_artifacts", SOURCE_DIR / "audit_historical_artifacts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HistoricalInventoryTests(unittest.TestCase):
    def test_inventory_hashes_only_relevant_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "summary.json").write_bytes(b"summary")
            (root / "report.pdf").write_bytes(b"pdf")
            (root / "ignore.csv").write_bytes(b"ignored")
            result = module.inventory(root)
        artifacts = {Path(row["path"]).name: row for row in result["artifacts"]}
        self.assertEqual(set(artifacts), {"summary.json", "report.pdf"})
        self.assertEqual(artifacts["summary.json"]["sha256"], hashlib.sha256(b"summary").hexdigest())

    def test_empty_or_missing_root_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "No historical artifacts"):
                module.inventory(directory)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                module.inventory(Path(directory) / "missing")


if __name__ == "__main__":
    unittest.main()
