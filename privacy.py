import math
import secrets

import numpy as np
import torch
from opacus.accountants.utils import get_noise_multiplier


def validate_embedding_privacy_parameters(epsilon, delta, num_tokens, num_examples):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Embedding epsilon must be a positive finite number.")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("Embedding delta must be between 0 and 1.")
    if num_tokens <= 0:
        raise ValueError("The fixed token vocabulary must not be empty.")
    if num_examples <= 0:
        raise ValueError("The private training dataset must not be empty.")
    if delta >= 1 / num_examples:
        raise ValueError(
            f"Embedding delta ({delta}) must be less than 1/N "
            f"({1 / num_examples}) for N={num_examples}."
        )


def calculate_embedding_clip_norm(pretrained_embedding_weight, quantile=0.95):
    if not math.isfinite(quantile) or not 0 < quantile <= 1:
        raise ValueError("Embedding clipping quantile must be in (0, 1].")
    if pretrained_embedding_weight.ndim != 2:
        raise ValueError("Pretrained word embeddings must be a two-dimensional table.")

    norms = torch.linalg.vector_norm(
        pretrained_embedding_weight.detach().to(device="cpu", dtype=torch.float64),
        ord=2,
        dim=1,
    )
    clip_norm = float(torch.quantile(norms, quantile).item())
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("Could not derive a positive finite embedding clipping norm.")
    return clip_norm


def clip_embedding_rows(embedding_table, clip_norm):
    if embedding_table.ndim != 2 or not embedding_table.is_floating_point():
        raise ValueError("Embedding table must be a two-dimensional floating-point tensor.")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("Embedding clipping norm must be a positive finite number.")

    row_norms = torch.linalg.vector_norm(embedding_table, ord=2, dim=1, keepdim=True)
    safe_norms = row_norms.clamp_min(torch.finfo(embedding_table.dtype).tiny)
    scales = (clip_norm / safe_norms).clamp(max=1.0)
    return embedding_table * scales


def embedding_noise_multiplier(epsilon, delta, num_tokens, num_examples):
    validate_embedding_privacy_parameters(
        epsilon=epsilon,
        delta=delta,
        num_tokens=num_tokens,
        num_examples=num_examples,
    )
    return get_noise_multiplier(
        target_epsilon=float(epsilon),
        target_delta=float(delta),
        sample_rate=1.0,
        steps=int(num_tokens),
        accountant="rdp",
    )


def _secure_standard_normal_values(count):
    random_bits = np.frombuffer(secrets.token_bytes(count * 8), dtype=np.uint64)
    uniform_values = ((random_bits >> np.uint64(11)).astype(np.float64) + 0.5) / (2**53)
    uniform_tensor = torch.from_numpy(uniform_values)
    return torch.erfinv(uniform_tensor.mul(2).sub(1)).mul(math.sqrt(2))


def add_secure_gaussian_noise(embedding_table, standard_deviation, rows_per_chunk=512):
    if embedding_table.ndim != 2 or not embedding_table.is_floating_point():
        raise ValueError("Embedding table must be a two-dimensional floating-point tensor.")
    if not math.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Gaussian noise standard deviation must be positive and finite.")
    if rows_per_chunk <= 0:
        raise ValueError("Noise chunk size must be positive.")

    noised_table = embedding_table.detach().to(device="cpu").clone()
    for start in range(0, noised_table.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, noised_table.shape[0])
        shape = (end - start, noised_table.shape[1])
        noise = _secure_standard_normal_values(math.prod(shape))
        noise = noise.reshape(shape).mul(standard_deviation).to(noised_table.dtype)
        noised_table[start:end].add_(noise)
    return noised_table


def privatize_embedding_table(
    clipped_teacher_mean,
    num_teachers,
    clip_norm,
    epsilon,
    delta,
    num_examples,
):
    if clipped_teacher_mean.ndim != 2 or not clipped_teacher_mean.is_floating_point():
        raise ValueError("Teacher mean must be a two-dimensional floating-point tensor.")
    if num_teachers <= 0:
        raise ValueError("At least one teacher is required.")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("Embedding clipping norm must be a positive finite number.")

    num_tokens = int(clipped_teacher_mean.shape[0])
    multiplier = embedding_noise_multiplier(
        epsilon=epsilon,
        delta=delta,
        num_tokens=num_tokens,
        num_examples=num_examples,
    )
    sensitivity = 2.0 * clip_norm / num_teachers
    noise_std = multiplier * sensitivity
    private_table = add_secure_gaussian_noise(clipped_teacher_mean, noise_std)

    metadata = {
        "mechanism": "per-token Gaussian mechanism with RDP composition",
        "privacy_unit": "one training example, replacement adjacency",
        "privacy_scope": "embedding table only; downstream DP-SGD excluded",
        "epsilon": float(epsilon),
        "delta": float(delta),
        "num_private_examples": int(num_examples),
        "num_tokens": num_tokens,
        "num_teachers": int(num_teachers),
        "clip_norm": float(clip_norm),
        "sensitivity_per_token": float(sensitivity),
        "noise_multiplier": float(multiplier),
        "noise_std": float(noise_std),
        "accountant": "Opacus RDP; sample_rate=1; steps=num_tokens",
    }
    return private_table, metadata
