"""Fingerprints for the embedding arrays used by the attack."""

import hashlib
import json
import math
import os

import numpy as np


def array_sha256(array):
    array = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header + b"\n")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sentence_selection_sha256(indices, sentences):
    digest = hashlib.sha256()
    for index in indices:
        entry = json.dumps([int(index), sentences[index]], ensure_ascii=True, separators=(",", ":"))
        digest.update(entry.encode("ascii") + b"\n")
    return digest.hexdigest()


def decoder_checkpoint_provenance(checkpoint_dir):
    config_path = os.path.abspath(os.path.join(checkpoint_dir, "training_args_and_best_models.json"))
    with open(config_path, "rb") as f:
        config_bytes = f.read()
    config = json.loads(config_bytes)
    if not config["best_models"]:
        raise ValueError("No best decoder checkpoint found.")
    checkpoint_path = os.path.abspath(config["best_models"][0][-1])
    return {
        "config_path": config_path,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }


def verify_distinct_conditions(conditions):
    if not conditions:
        raise ValueError("No embedding conditions provided.")

    reference_ids = None
    table_owners = {}
    for label, provenance in conditions.items():
        ids_hash = provenance["token_ids_array_sha256"]
        if reference_ids is None:
            reference_ids = ids_hash
        elif ids_hash != reference_ids:
            raise ValueError(f"Token IDs differ for condition {label}.")

        table_hash = provenance["table_array_sha256"]
        if table_hash in table_owners:
            raise ValueError(
                f"Identical embedding tables for {table_owners[table_hash]} and {label}."
            )
        table_owners[table_hash] = label


def verify_expected_provenance(provenance, expected_table, expected_token_ids, expected_export=None):
    for key, expected in (
        ("table_array_sha256", expected_table),
        ("token_ids_array_sha256", expected_token_ids),
        ("export_provenance_sha256", expected_export),
    ):
        if expected is not None and provenance[key] != expected:
            raise ValueError(f"Loaded {key} does not match the preflight.")


def verify_dped_research(provenance, epsilon):
    metadata = provenance.get("export_metadata")
    if metadata is None:
        raise ValueError("DPED research requires export metadata.txt.")
    values = metadata["values"]
    required = (
        "epsilon", "delta", "num_teachers", "clip_norm", "sensitivity_per_token",
        "noise_multiplier", "noise_std", "num_tokens", "num_private_examples",
    )
    for key in required:
        if key not in values:
            raise ValueError(f"DPED export metadata is missing {key}.")
        try:
            value = float(values[key])
        except ValueError as exc:
            raise ValueError(f"Invalid DPED metadata value for {key}.") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid DPED metadata value for {key}.")
    if epsilon is None or float(epsilon) != float(values["epsilon"]):
        raise ValueError("DPED epsilon label does not match export metadata.")
    files = provenance.get("generation_files", {})
    for filename in (
        "generation_config.json", "generation_provenance.json",
        "generation_code.zip", "generation_requirements.observed.txt",
    ):
        if filename not in files:
            raise ValueError(f"DPED research requires generation artifact {filename}.")
    for filename, hash_key in (
        ("token_embeddings.npy", "table_file_sha256"),
        ("token_ids.npy", "token_ids_file_sha256"),
    ):
        recorded = values.get(filename + "_sha256")
        if recorded is None or recorded != provenance[hash_key]:
            raise ValueError(f"DPED export hash missing or mismatched for {filename}.")
    config = provenance["generation_config"]
    if (float(config["epsilon"]) != float(values["epsilon"])
            or int(config["num_teachers"]) != int(values["num_teachers"])):
        raise ValueError("DPED generation configuration differs from export metadata.")
    if int(values["num_tokens"]) != provenance["table_shape"][0]:
        raise ValueError("DPED vocabulary size differs from export metadata.")
    manifest = provenance["generation_manifest"]
    for filename in ("generation_code.zip", "generation_requirements.observed.txt"):
        if manifest["files"].get(filename) != files[filename]:
            raise ValueError(f"DPED generation artifact hash differs for {filename}.")
    arguments = manifest["arguments"]
    original_config = arguments["config"]
    for key, value in config.items():
        if key == "epsilon" and arguments.get("embedding_privacy_sweep"):
            continue
        if key not in original_config or original_config[key] != value:
            raise ValueError(f"DPED generation configuration changed for {key}.")
    for key in original_config:
        if key not in config:
            raise ValueError(f"DPED generation configuration is missing {key}.")
