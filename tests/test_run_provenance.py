import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "run_provenance.py"
)
spec = importlib.util.spec_from_file_location("run_provenance", MODULE_PATH)
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)


@unittest.skipUnless(shutil.which("git"), "git is unavailable")
class GitProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "core.autocrlf", "false")
        (self.root / "src").mkdir()
        (self.root / "src" / "attack.py").write_text("original\n", encoding="utf-8")
        (self.root / "src" / "deleted.py").write_text("delete\n", encoding="utf-8")
        (self.root / ".gitignore").write_text("src/ignored.py\n", encoding="utf-8")
        self.git("add", ".")
        self.git("-c", "core.hooksPath=", "commit", "-qm", "initial")
        self.revision = self.git("rev-parse", "HEAD").strip()

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.decode("utf-8")

    def test_selected_code_includes_changes_untracked_and_deleted(self):
        changed = self.root / "src" / "attack.py"
        changed.write_bytes(b"modified\n")
        (self.root / "src" / "new.py").write_bytes(b"new\n")
        (self.root / "src" / "ignored.py").write_bytes(b"ignored\n")
        (self.root / "src" / "private.csv").write_bytes(b"private data")
        (self.root / "private.py").write_bytes(b"not selected")
        self.git("rm", "-q", "src/deleted.py")

        result = provenance.code_provenance(self.root, ["src", "missing.py"])
        self.assertEqual(result["revision"], self.revision)
        self.assertTrue(result["tracked_dirty"])
        files = result["files"]
        self.assertEqual(set(files), {
            "src/attack.py", "src/deleted.py", "src/new.py", "missing.py",
        })
        self.assertEqual(files["src/attack.py"]["sha256"], hashlib.sha256(b"modified\n").hexdigest())
        self.assertEqual(files["src/attack.py"]["git_status"], " M")
        self.assertTrue(files["src/attack.py"]["tracked"])
        self.assertEqual(files["src/deleted.py"]["git_status"], "D ")
        self.assertFalse(files["src/deleted.py"]["exists"])
        self.assertIsNone(files["src/deleted.py"]["sha256"])
        for name in ("src/new.py",):
            self.assertTrue(files[name]["untracked"])
            self.assertIsNotNone(files[name]["sha256"])
        self.assertFalse(files["missing.py"]["exists"])
        self.assertIsNone(files["missing.py"]["git_status"])

    def test_clean_and_noncode_change(self):
        result = provenance.code_provenance(self.root, ["src"])
        self.assertFalse(result["tracked_dirty"])
        self.assertEqual(result["files"]["src/attack.py"]["git_status"], "  ")
        first_hash = result["code_tree_sha256"]
        (self.root / ".gitignore").write_text("changed\n", encoding="utf-8")
        result = provenance.code_provenance(self.root, ["src"])
        self.assertTrue(result["tracked_dirty"])
        self.assertNotIn(".gitignore", result["files"])
        self.assertEqual(first_hash, result["code_tree_sha256"])
        (self.root / "src" / "attack.py").write_text("changed\n", encoding="utf-8")
        self.assertNotEqual(first_hash, provenance.code_provenance(self.root, ["src"])["code_tree_sha256"])

    def test_outside_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            provenance.code_provenance(self.root, ["../outside.py"])

    def test_archive_contains_fingerprinted_code_and_rejects_changes(self):
        result = provenance.code_provenance(self.root, ["src"])
        archive = self.root / "code.zip"
        provenance.write_code_archive(self.root, result, archive)
        with zipfile.ZipFile(archive) as source:
            self.assertEqual(source.read("src/attack.py"), (self.root / "src" / "attack.py").read_bytes())
        (self.root / "src" / "attack.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Code changed"):
            provenance.write_code_archive(self.root, result, archive)

    def test_missing_repository_and_unborn_head_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                provenance.code_provenance(directory, ["."])
            self.git("init", "-q", str(Path(directory) / "empty"))
            with self.assertRaises(RuntimeError):
                provenance.code_provenance(Path(directory) / "empty", ["."])


class ProvenanceTests(unittest.TestCase):
    def test_missing_git_is_explicit(self):
        with mock.patch.object(provenance.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "Git executable"):
                provenance.code_provenance(".", ["."])

    def test_export_metadata_preserves_only_actual_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.txt"
            content = b"stage=after_dped\nshape=(5, 2)\nprivacy=a=b\n"
            path.write_bytes(content)
            result = provenance.export_metadata(path)
            self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(result["values"], {
                "stage": "after_dped", "shape": "(5, 2)", "privacy": "a=b",
            })
            path.write_text("epsilon=1\nepsilon=2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                provenance.export_metadata(path)
            with self.assertRaises(FileNotFoundError):
                provenance.export_metadata(path.with_name("missing.txt"))

    def test_snapshot_records_all_observed_versions(self):
        distributions = [
            mock.Mock(metadata={"Name": "Example"}, version="1.2.3"),
            mock.Mock(metadata={"Name": "other"}, version="0.4+local"),
            mock.Mock(metadata={"Name": "Example"}, version="2.0"),
        ]
        with mock.patch.object(provenance.importlib.metadata, "distributions", return_value=distributions):
            snapshot = provenance.environment_snapshot()
        self.assertEqual(snapshot["kind"], "observed_installed_distributions_not_resolved_lock")
        self.assertEqual(snapshot["distributions"], [
            {"name": "Example", "version": "1.2.3"},
            {"name": "Example", "version": "2.0"},
            {"name": "other", "version": "0.4+local"},
        ])
        self.assertTrue(snapshot["python"])

    def test_arguments_are_copied_without_inferred_defaults(self):
        args = argparse.Namespace(seed=7, input_paths=["a"])
        result = provenance.record_arguments(args)
        args.input_paths.append("b")
        self.assertEqual(result, {"seed": 7, "input_paths": ["a"]})


if __name__ == "__main__":
    unittest.main()
