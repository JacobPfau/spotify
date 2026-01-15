"""Active comparison selection based on expected information gain."""

from __future__ import annotations

import random as py_random
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jax import Array

from src.data import Artist, Comparison, Song, get_uncompared_pairs


def compute_entropy(probs: Array) -> float:
    """Compute entropy of a probability distribution."""
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    probs = jnp.clip(probs, eps, 1.0 - eps)
    return float(-jnp.sum(probs * jnp.log(probs)))


def compute_pair_score(
    samples: dict[str, Array],
    idx_a: int,
    idx_b: int,
) -> float:
    """Compute expected information gain score for a pair.

    score = P(comparable) * H(response distribution)

    Args:
        samples: Posterior samples from inference
        idx_a: Index of artist A
        idx_b: Index of artist B

    Returns:
        Score (higher = more informative)
    """
    genre_positions = samples["genre_positions"]  # (n_samples, n_artists, k_dims)
    utilities = samples["utilities"]  # (n_samples, n_artists)
    lambdas = samples["lambdas"]  # (n_samples, n_artists)
    alpha = samples["alpha"]  # (n_samples,)
    tau = samples["tau"]  # (n_samples, k_dims)

    # Get parameters for this pair across all samples
    g_a = genre_positions[:, idx_a, :]  # (n_samples, k_dims)
    g_b = genre_positions[:, idx_b, :]
    s_a = utilities[:, idx_a]  # (n_samples,)
    s_b = utilities[:, idx_b]
    lambda_a = lambdas[:, idx_a]
    lambda_b = lambdas[:, idx_b]

    # Compute comparability
    genre_diff_sq = jnp.sum(((g_a - g_b) / tau) ** 2, axis=-1)
    p_comparable = jax.nn.sigmoid(alpha - genre_diff_sq)

    # Compute preference probability given comparable
    noise_var = 1.0 / lambda_a + 1.0 / lambda_b
    z = (s_a - s_b) / jnp.sqrt(noise_var)
    p_a_given_comp = jax.scipy.stats.norm.cdf(z)

    # Posterior mean of outcome probabilities
    mean_p_comparable = float(jnp.mean(p_comparable))
    mean_p_a_given_comp = float(jnp.mean(p_a_given_comp))

    # Three-outcome probabilities (using posterior means)
    p_abstain = 1.0 - mean_p_comparable
    p_prefer_a = mean_p_comparable * mean_p_a_given_comp
    p_prefer_b = mean_p_comparable * (1.0 - mean_p_a_given_comp)

    # Entropy of response distribution
    probs = jnp.array([p_abstain, p_prefer_a, p_prefer_b])
    entropy = compute_entropy(probs)

    # Score: weight by comparability (don't waste user time on incomparable pairs)
    score = mean_p_comparable * entropy

    return score


def select_next_pair(
    artists: list[Artist],
    comparisons: list[Comparison],
    samples: dict[str, Array],
    artist_id_to_idx: dict[str, int],
    top_k: int = 3,
) -> tuple[str, str] | None:
    """Select the next pair to compare using active learning.

    Args:
        artists: List of artists
        comparisons: Existing comparisons
        samples: Posterior samples from inference
        artist_id_to_idx: Mapping from artist ID to index
        top_k: Sample from top-k pairs (adds randomization)

    Returns:
        Tuple of (artist_a_id, artist_b_id) or None if all pairs compared
    """
    uncompared = get_uncompared_pairs(artists, comparisons)

    if not uncompared:
        return None

    # Score all uncompared pairs
    scores = []
    for aid_a, aid_b in uncompared:
        idx_a = artist_id_to_idx[aid_a]
        idx_b = artist_id_to_idx[aid_b]
        score = compute_pair_score(samples, idx_a, idx_b)
        scores.append((score, aid_a, aid_b))

    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)

    # Sample from top-k to add randomization
    top_pairs = scores[: min(top_k, len(scores))]
    _, aid_a, aid_b = py_random.choice(top_pairs)

    return (aid_a, aid_b)


def select_random_pair(
    artists: list[Artist],
    comparisons: list[Comparison],
) -> tuple[str, str] | None:
    """Select a random uncompared pair (for Phase 1 / before inference).

    Args:
        artists: List of artists
        comparisons: Existing comparisons

    Returns:
        Tuple of (artist_a_id, artist_b_id) or None if all pairs compared
    """
    uncompared = get_uncompared_pairs(artists, comparisons)

    if not uncompared:
        return None

    return py_random.choice(uncompared)


def select_songs_for_comparison(
    artist_a_id: str,
    artist_b_id: str,
    songs_by_artist: dict[str, list[Song]],
) -> tuple[Song, Song]:
    """Select random songs from each artist for comparison.

    Args:
        artist_a_id: ID of artist A
        artist_b_id: ID of artist B
        songs_by_artist: Mapping from artist ID to songs

    Returns:
        Tuple of (song_a, song_b)
    """
    song_a = py_random.choice(songs_by_artist[artist_a_id])
    song_b = py_random.choice(songs_by_artist[artist_b_id])
    return song_a, song_b
