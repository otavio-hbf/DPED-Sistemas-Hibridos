"""Local, observed provenance for attack runs; no information is transmitted."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import zipfile


def _git(directory, *args, allowed=(0,)):
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(directory), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git executable is unavailable") from exc
    if result.returncode not in allowed:
        raise RuntimeError(f"Git command failed: {args[0]} (exit {result.returncode})")
    return result


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_provenance(repository, code_paths):
    """Fingerprint selected Python files/directories relative to the repository root.

    Directories are searched recursively for tracked and untracked .py files.
    Select source directories narrowly, not data or artifact
    directories. Git and a committed HEAD are required. No global status paths
    are collected; tracked_dirty covers the entire repository, whereas file
    statuses and untracked files cover only the selection.
    """
    root = Path(os.fsdecode(_git(repository, "rev-parse", "--show-toplevel").stdout.rstrip(b"\r\n")))
    revision = _git(root, "rev-parse", "--verify", "HEAD").stdout.decode("ascii").strip()
    if not code_paths:
        raise ValueError("Select at least one source file or directory")

    pathspecs = []
    requested_files = set()
    for selected in code_paths:
        path = Path(selected)
        path = path if path.is_absolute() else root / path
        # Do not resolve the final component: symlinks must not be followed.
        relative = path.absolute().relative_to(root.absolute())
        if ".." in relative.parts:
            raise ValueError("Code paths must be inside the repository")
        name = relative.as_posix()
        if any((root / Path(*relative.parts[:i])).is_symlink()
               for i in range(1, len(relative.parts) + 1)):
            raise ValueError("Symlinked code paths are not supported")
        if path.is_dir():
            prefix = "" if name == "." else name + "/"
            pathspecs.append(f":(top,glob){prefix}**/*.py")
        elif path.suffix == ".py":
            requested_files.add(name)
            pathspecs.append(f":(top,literal){name}")
        else:
            raise ValueError("Code paths must be Python files or existing directories")

    # ls-files includes deleted tracked files; pathspecs restrict selected code.
    files = _git(root, "ls-files", "-z", "--cached", "--", *pathspecs)
    tracked_names = {os.fsdecode(name) for name in files.stdout.split(b"\0") if name}
    untracked = _git(root, "ls-files", "-z", "--others", "--exclude-standard", "--", *pathspecs)
    untracked_names = {os.fsdecode(name) for name in untracked.stdout.split(b"\0") if name}
    names = tracked_names | untracked_names | requested_files
    status = _git(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=no",
        "--no-renames", "--", *pathspecs,
    )
    changes = {}
    for entry in status.stdout.split(b"\0"):
        if entry:
            changes[os.fsdecode(entry[3:])] = entry[:2].decode("ascii")
    names.update(changes)

    entries = {}
    for name in sorted(names):
        path = root / name
        # A symlink in any component could point to a data file outside the tree.
        if any((root / Path(*Path(name).parts[:i])).is_symlink()
               for i in range(1, len(Path(name).parts) + 1)):
            raise ValueError(f"Symlink in selected code path: {name}")
        entries[name] = {
            "sha256": _file_sha256(path) if path.is_file() else None,
            "exists": path.is_file(),
            "git_status": changes.get(name, "  ") if name in tracked_names or name in changes else None,
            "tracked": name in tracked_names,
            "untracked": name in untracked_names,
        }

    unstaged = _git(root, "diff", "--quiet", "--ignore-submodules=none", "--", allowed=(0, 1))
    staged = _git(root, "diff", "--cached", "--quiet", "--ignore-submodules=none", "--", allowed=(0, 1))
    tree = {name: info["sha256"] for name, info in entries.items()}
    tree_bytes = json.dumps(tree, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "revision": revision,
        "tracked_dirty": unstaged.returncode == 1 or staged.returncode == 1,
        "code_tree_sha256": hashlib.sha256(tree_bytes).hexdigest(),
        "files": entries,
    }


def environment_snapshot():
    """Return installed distribution versions, not a resolved reproducible lock."""
    packages = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        version = dist.version
        if not name or not version:
            raise ValueError("Installed distribution has no name or version")
        packages.append({"name": name, "version": version})
    packages.sort(key=lambda item: (item["name"].lower(), item["version"]))
    return {
        "kind": "observed_installed_distributions_not_resolved_lock",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "distributions": packages,
    }


def export_metadata(path):
    """Read an export's key=value metadata without inferring missing values."""
    path = Path(path)
    content = path.read_bytes()
    values = {}
    for line in content.decode("utf-8").splitlines():
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError("Malformed or duplicate export metadata key")
        values[key] = value
    return {"sha256": hashlib.sha256(content).hexdigest(), "values": values}


def record_arguments(arguments):
    """Copy received mapping or argparse Namespace; caller must exclude secrets."""
    if isinstance(arguments, argparse.Namespace):
        arguments = vars(arguments)
    return copy.deepcopy(dict(arguments))


def project_code_provenance(repository):
    root = Path(repository)
    paths = [
        "config.py", "data_loader.py", "main.py", "models.py", "privacy.py",
        "training.py", "utils.py", "run_provenance.py",
        os.path.join("algen", "SIH-ALGEN-main", "src"),
    ]
    return code_provenance(root, paths)


def write_code_archive(repository, provenance, destination):
    root = Path(repository)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, info in provenance["files"].items():
            if not info["exists"]:
                continue
            content = (root / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != info["sha256"]:
                raise RuntimeError(f"Code changed while creating snapshot: {name}")
            archive.writestr(name, content)


def write_environment_versions(snapshot, destination):
    with open(destination, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Observed installed versions; not a resolved, hash-verified lock.\n")
        for distribution in snapshot["distributions"]:
            f.write(f"{distribution['name']}=={distribution['version']}\n")


def capture_execution(repository, arguments):
    return {
        "code": project_code_provenance(repository),
        "environment": environment_snapshot(),
        "arguments": record_arguments(arguments),
    }


def write_execution(repository, provenance, output_dir, prefix):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / f"{prefix}_code.zip"
    requirements_path = output / f"{prefix}_requirements.observed.txt"
    write_code_archive(repository, provenance["code"], archive_path)
    write_environment_versions(provenance["environment"], requirements_path)
    recorded = dict(provenance)
    recorded["files"] = {
        archive_path.name: _file_sha256(archive_path),
        requirements_path.name: _file_sha256(requirements_path),
    }
    with (output / f"{prefix}_provenance.json").open("w", encoding="utf-8") as f:
        json.dump(recorded, f, indent=2, sort_keys=True, allow_nan=False)
