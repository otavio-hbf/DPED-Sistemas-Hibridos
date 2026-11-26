"""Inventory historical experiment artifacts without changing them."""

import argparse
import json
import os
from pathlib import Path

from embedding_provenance import file_sha256


def inventory(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {root}")
    filenames = {"summary.json", "sentence_results.csv", "token_embeddings.npy", "token_ids.npy", "metadata.txt"}
    artifacts = []
    for directory, _, files in os.walk(root):
        for name in sorted(files):
            if name in filenames or name.lower().endswith(".pdf"):
                path = Path(directory) / name
                if path.is_symlink():
                    raise ValueError(f"Symlinked artifact is not supported: {path}")
                artifacts.append({
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                })
    if not artifacts:
        raise FileNotFoundError(f"No historical artifacts found under {root}")
    return {"root": str(root.resolve()), "artifacts": artifacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory containing historical artifacts.")
    parser.add_argument("--output", required=True, help="New manifest path; existing files are not overwritten.")
    args = parser.parse_args()
    result = inventory(args.root)
    with open(args.output, "x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Inventoried {len(result['artifacts'])} artifacts in {args.output}; verify their origin and completeness.")


if __name__ == "__main__":
    main()
