"""Offline bandwidth and Nyström construction for a frozen ViRDM reference."""

from __future__ import annotations

import torch


def gamma_from_sigma(sigma: float) -> float:
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return 1.0 / (2.0 * sigma * sigma)


@torch.no_grad()
def median_bandwidth(
    features: torch.Tensor,
    *,
    max_subsample: int = 2000,
    scale: float = 1.0,
) -> float:
    """Match the original strict-upper-triangle median-distance estimator."""

    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("median bandwidth requires at least two feature rows")
    count = min(int(max_subsample), int(features.shape[0]))
    indices = torch.randperm(features.shape[0], device=features.device)[:count]
    points = features[indices]
    distance_squared = torch.cdist(points, points).pow(2)
    upper = torch.triu(
        torch.ones(
            count, count, dtype=torch.bool, device=distance_squared.device
        ),
        diagonal=1,
    )
    sigma = distance_squared[upper].median().sqrt().item()
    return max(float(sigma), 1.0e-5) * float(scale)


@torch.no_grad()
def kmeans_landmarks(
    features: torch.Tensor,
    n_landmarks: int,
    *,
    iterations: int = 20,
    seed: int = 0,
    chunk: int = 16384,
) -> torch.Tensor:
    device = features.device
    row_count = int(features.shape[0])
    if not 1 <= int(n_landmarks) <= row_count:
        raise ValueError(
            f"n_landmarks must be in [1,{row_count}], got {n_landmarks}"
        )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    centers = features[
        torch.randperm(row_count, generator=generator, device=device)[:n_landmarks]
    ].clone()
    for _ in range(int(iterations)):
        labels = torch.empty(row_count, dtype=torch.long, device=device)
        center_norm = (centers * centers).sum(1)
        for start in range(0, row_count, int(chunk)):
            rows = features[start : start + int(chunk)]
            distance_squared = (
                (rows * rows).sum(1)[:, None]
                + center_norm[None, :]
                - 2.0 * (rows @ centers.T)
            )
            labels[start : start + rows.shape[0]] = distance_squared.argmin(1)
        next_centers = torch.zeros_like(centers)
        counts = torch.zeros(n_landmarks, device=device)
        next_centers.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones(row_count, device=device))
        empty = counts == 0
        next_centers = next_centers / counts.clamp_min(1.0)[:, None]
        if empty.any():
            replacements = torch.randperm(
                row_count, generator=generator, device=device
            )[: int(empty.sum())]
            next_centers[empty] = features[replacements]
        centers = next_centers
    return centers


@torch.no_grad()
def streaming_reference_mean(
    landmarks: torch.Tensor,
    landmark_norm: torch.Tensor,
    pool: torch.Tensor,
    gamma: float,
    *,
    chunk: int = 50000,
) -> torch.Tensor:
    mean = torch.zeros(
        landmarks.shape[0], dtype=torch.float64, device=landmarks.device
    )
    for start in range(0, pool.shape[0], int(chunk)):
        rows = pool[start : start + int(chunk)].double()
        row_norm = (rows * rows).sum(1)
        distance_squared = (
            landmark_norm[:, None]
            + row_norm[None, :]
            - 2.0 * (landmarks @ rows.T)
        ).clamp_min(0)
        mean += torch.exp(-float(gamma) * distance_squared).sum(1)
    return mean / pool.shape[0]


@torch.no_grad()
def reference_self_kernel_mean(
    pool: torch.Tensor,
    sigma: float,
    *,
    n_subsample: int = 100000,
    row_chunk: int = 4096,
    column_chunk: int = 50000,
) -> float:
    rows = pool[: min(int(n_subsample), pool.shape[0])].float()
    gamma = gamma_from_sigma(sigma)
    row_norm = (rows * rows).sum(1)
    total = rows.new_zeros(())
    for row_start in range(0, rows.shape[0], int(row_chunk)):
        left = rows[row_start : row_start + int(row_chunk)]
        left_norm = (left * left).sum(1)
        for column_start in range(0, rows.shape[0], int(column_chunk)):
            right = rows[column_start : column_start + int(column_chunk)]
            right_norm = row_norm[column_start : column_start + int(column_chunk)]
            distance_squared = (
                left_norm[:, None]
                + right_norm[None, :]
                - 2.0 * (left @ right.T)
            ).clamp_min(0)
            total += torch.exp(-gamma * distance_squared).sum()
    return float(total / (rows.shape[0] * rows.shape[0]))


@torch.no_grad()
def build_nystrom_reference(
    pool: torch.Tensor,
    sigma: float,
    *,
    n_landmarks: int = 4096,
    fit_n: int = 200000,
    kmeans_iterations: int = 20,
    krr_n: int = 100000,
    seed: int = 0,
    jitter: float = 1.0e-6,
    device: torch.device | str | None = None,
) -> dict:
    """Build the exact frozen bundle used by the ViRDM training loss."""

    if pool.ndim != 2:
        raise ValueError(f"feature pool must be [N,D], got {tuple(pool.shape)}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    working = pool.to(device=device, dtype=torch.float32)
    row_count, feature_dim = working.shape
    if int(n_landmarks) > row_count:
        raise ValueError(
            f"n_landmarks {n_landmarks} exceeds pool rows {row_count}"
        )
    generator = torch.Generator(device=working.device).manual_seed(int(seed))
    fit_indices = torch.randperm(
        row_count, generator=generator, device=working.device
    )[: min(int(fit_n), row_count)]
    fit = working[fit_indices]
    landmarks = kmeans_landmarks(
        fit,
        int(n_landmarks),
        iterations=int(kmeans_iterations),
        seed=int(seed),
    ).double()
    landmark_norm = (landmarks * landmarks).sum(1)
    distance_squared = (
        landmark_norm[:, None]
        + landmark_norm[None, :]
        - 2.0 * (landmarks @ landmarks.T)
    ).clamp_min(0)
    kernel = torch.exp(-gamma_from_sigma(sigma) * distance_squared)
    kernel += float(jitter) * torch.eye(
        n_landmarks, dtype=torch.float64, device=working.device
    )
    cholesky = torch.linalg.cholesky(kernel)
    reference_mean = streaming_reference_mean(
        landmarks,
        landmark_norm,
        working,
        gamma_from_sigma(sigma),
    )
    alpha = torch.cholesky_solve(reference_mean[:, None], cholesky).squeeze(1)
    k_rr = reference_self_kernel_mean(
        working, sigma, n_subsample=min(int(krr_n), row_count)
    )
    return {
        "Z": landmarks.float().cpu(),
        "alpha": alpha.float().cpu(),
        "sigma": float(sigma),
        "k_rr": float(k_rr),
        "M": int(n_landmarks),
        "d_in": int(feature_dim),
    }
