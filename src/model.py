"""NumPyro Bayesian model for artist preference learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

if TYPE_CHECKING:
    from jax import Array

from src.data import Comparison, ComparisonResult


def artist_preference_model(
    n_artists: int,
    artist_a_idx: Array,
    artist_b_idx: Array,
    outcomes: Array | None = None,
    prior_genre_means: Array | None = None,
    k_dims: int = 1,
) -> None:
    """NumPyro model for artist preference learning.

    Args:
        n_artists: Number of artists
        artist_a_idx: Array of artist A indices for each comparison
        artist_b_idx: Array of artist B indices for each comparison
        outcomes: Array of outcomes (0=abstain, 1=prefer A, 2=prefer B), or None for prior predictive
        prior_genre_means: Prior means for genre positions from LLM (shape: n_artists x k_dims)
        k_dims: Number of genre dimensions (1 or 2)
    """
    # Set up prior means for genre positions
    if prior_genre_means is None:
        prior_genre_means = jnp.zeros((n_artists, k_dims))

    # === Priors ===

    # Comparability intercept: baseline comparability at zero distance
    alpha = numpyro.sample("alpha", dist.Normal(1.0, 1.0))

    # Comparability lengthscales (ARD for k_dims > 1)
    tau = numpyro.sample("tau", dist.HalfNormal(1.0).expand([k_dims]))

    # Utility scale
    sigma_s = numpyro.sample("sigma_s", dist.HalfNormal(1.0))

    # Genre positions: first artist anchored at 0
    # Sample non-anchor artists with independent normals per dimension
    genre_raw = numpyro.sample(
        "genre_raw",
        dist.Normal(prior_genre_means[1:], 1.0).to_event(1),
    )
    # genre_raw has shape (n_artists - 1, k_dims)

    # Anchor first artist at origin
    genre_anchor = jnp.zeros((1, k_dims))
    genre_positions = numpyro.deterministic(
        "genre_positions", jnp.concatenate([genre_anchor, genre_raw], axis=0)
    )

    # Artist utilities: first artist anchored at 0
    utility_raw = numpyro.sample(
        "utility_raw",
        dist.Normal(jnp.zeros(n_artists - 1), sigma_s).to_event(1),
    )

    utility_anchor = jnp.array([0.0])
    utilities = numpyro.deterministic(
        "utilities", jnp.concatenate([utility_anchor, utility_raw])
    )

    # Artist consistency (lambda): higher = more consistent song quality
    lambdas = numpyro.sample(
        "lambdas",
        dist.Gamma(2.0, 1.0).expand([n_artists]).to_event(1),
    )

    # === Likelihood ===

    if artist_a_idx is not None and len(artist_a_idx) > 0:
        # Get parameters for each comparison
        g_a = genre_positions[artist_a_idx]  # (n_comparisons, k_dims)
        g_b = genre_positions[artist_b_idx]

        s_a = utilities[artist_a_idx]  # (n_comparisons,)
        s_b = utilities[artist_b_idx]

        lambda_a = lambdas[artist_a_idx]
        lambda_b = lambdas[artist_b_idx]

        # Compute comparability probability
        # P(comparable) = sigmoid(alpha - sum_k (g_a_k - g_b_k)^2 / tau_k^2)
        genre_diff_sq = jnp.sum(((g_a - g_b) / tau) ** 2, axis=-1)
        p_comparable = jax.nn.sigmoid(alpha - genre_diff_sq)

        # Compute preference probability given comparable
        # P(A > B | comparable) = Phi((s_a - s_b) / sqrt(1/lambda_a + 1/lambda_b))
        noise_var = 1.0 / lambda_a + 1.0 / lambda_b
        z = (s_a - s_b) / jnp.sqrt(noise_var)
        p_a_given_comp = jax.scipy.stats.norm.cdf(z)

        # Three-outcome probabilities
        p_abstain = 1.0 - p_comparable
        p_prefer_a = p_comparable * p_a_given_comp
        p_prefer_b = p_comparable * (1.0 - p_a_given_comp)

        # Stack into categorical probabilities
        probs = jnp.stack([p_abstain, p_prefer_a, p_prefer_b], axis=-1)

        # Observe outcomes
        with numpyro.plate("comparisons", len(artist_a_idx)):
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=outcomes)


# Need to import jax at module level for the model
import jax


def encode_comparisons(
    comparisons: list[Comparison],
    artist_id_to_idx: dict[str, int],
) -> tuple[Array, Array, Array]:
    """Encode comparisons into arrays for the model.

    Returns:
        artist_a_idx: Array of artist A indices
        artist_b_idx: Array of artist B indices
        outcomes: Array of outcomes (0=abstain, 1=prefer A, 2=prefer B)
    """
    artist_a_idx = []
    artist_b_idx = []
    outcomes = []

    for c in comparisons:
        artist_a_idx.append(artist_id_to_idx[c.artist_a_id])
        artist_b_idx.append(artist_id_to_idx[c.artist_b_id])

        if c.result == ComparisonResult.ABSTAIN:
            outcomes.append(0)
        elif c.result == ComparisonResult.PREFER_A:
            outcomes.append(1)
        else:
            outcomes.append(2)

    return (
        jnp.array(artist_a_idx),
        jnp.array(artist_b_idx),
        jnp.array(outcomes),
    )


def run_inference(
    n_artists: int,
    comparisons: list[Comparison],
    artist_id_to_idx: dict[str, int],
    prior_genre_means: Array | None = None,
    k_dims: int = 1,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 2,
    seed: int = 0,
) -> dict[str, Array]:
    """Run MCMC inference on the artist preference model.

    Args:
        n_artists: Number of artists
        comparisons: List of comparisons
        artist_id_to_idx: Mapping from artist ID to index
        prior_genre_means: Prior means for genre positions (n_artists x k_dims)
        k_dims: Number of genre dimensions
        num_warmup: Number of warmup samples
        num_samples: Number of posterior samples
        num_chains: Number of MCMC chains
        seed: Random seed

    Returns:
        Dictionary of posterior samples
    """
    # Encode comparisons
    artist_a_idx, artist_b_idx, outcomes = encode_comparisons(
        comparisons, artist_id_to_idx
    )

    # Set up NUTS sampler
    kernel = NUTS(artist_preference_model)

    # Run MCMC
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    # Use time-based seed variation to avoid JAX caching issues
    import time
    actual_seed = seed + int(time.time() * 1000) % 10000
    rng_key = random.PRNGKey(actual_seed)
    mcmc.run(
        rng_key,
        n_artists=n_artists,
        artist_a_idx=artist_a_idx,
        artist_b_idx=artist_b_idx,
        outcomes=outcomes,
        prior_genre_means=prior_genre_means,
        k_dims=k_dims,
    )

    return mcmc.get_samples()


def extract_posterior_summary(
    samples: dict[str, Array],
    artist_ids: list[str],
) -> dict[str, dict[str, float]]:
    """Extract posterior summary statistics for each artist.

    Returns:
        Dictionary with artist_id keys containing:
        - utility_mean, utility_std
        - genre_mean (1D array), genre_std
        - lambda_mean, lambda_std
    """
    utilities = samples["utilities"]
    genre_positions = samples["genre_positions"]
    lambdas = samples["lambdas"]

    summary = {}
    for i, aid in enumerate(artist_ids):
        summary[aid] = {
            "utility_mean": float(jnp.mean(utilities[:, i])),
            "utility_std": float(jnp.std(utilities[:, i])),
            "genre_mean": jnp.mean(genre_positions[:, i, :], axis=0).tolist(),
            "genre_std": jnp.std(genre_positions[:, i, :], axis=0).tolist(),
            "lambda_mean": float(jnp.mean(lambdas[:, i])),
            "lambda_std": float(jnp.std(lambdas[:, i])),
        }

    return summary


def compute_ranking_confidence(
    n_comparisons: int,
    total_pairs: int,
) -> dict[str, str | float | int]:
    """Compute ranking confidence based on comparison progress.

    Shows progress through ranking stages without misleading accuracy numbers.
    Accuracy is only shown once we have enough data to be meaningful.

    Returns:
        {
            "percent_complete": float (0-100),
            "status": str ("warming_up" | "early" | "refining" | "solid"),
            "status_label": str (human-readable),
            "comparisons_to_next": int,
            "show_accuracy": bool,
        }
    """
    if total_pairs == 0:
        return {
            "percent_complete": 0.0,
            "status": "warming_up",
            "status_label": "Warming up",
            "comparisons_to_next": 10,
            "show_accuracy": False,
        }

    percent = (n_comparisons / total_pairs) * 100

    # Stages based on % of pairs compared
    # Don't show accuracy until we have meaningful data (after first inference)
    if n_comparisons < 10:
        status = "warming_up"
        status_label = "Warming up"
        next_threshold = 10
        show_accuracy = False
    elif percent < 25:
        status = "early"
        status_label = "Early ranking"
        next_threshold = int(0.25 * total_pairs)
        show_accuracy = False
    elif percent < 50:
        status = "refining"
        status_label = "Refining"
        next_threshold = int(0.5 * total_pairs)
        show_accuracy = True
    else:
        status = "solid"
        status_label = "Solid ranking"
        next_threshold = total_pairs
        show_accuracy = True

    comparisons_to_next = max(0, next_threshold - n_comparisons)

    return {
        "percent_complete": round(percent, 1),
        "status": status,
        "status_label": status_label,
        "comparisons_to_next": comparisons_to_next,
        "show_accuracy": show_accuracy,
    }


def compute_comparability_matrix(
    samples: dict[str, Array],
    n_artists: int,
) -> Array:
    """Compute posterior mean comparability between all artist pairs.

    Returns:
        n_artists x n_artists matrix of comparability probabilities
    """
    genre_positions = samples["genre_positions"]  # (n_samples, n_artists, k_dims)
    alpha = samples["alpha"]  # (n_samples,)
    tau = samples["tau"]  # (n_samples, k_dims)

    n_samples = genre_positions.shape[0]

    # Compute pairwise comparability for each sample
    comparability = jnp.zeros((n_artists, n_artists))

    for i in range(n_artists):
        for j in range(n_artists):
            if i == j:
                continue

            g_i = genre_positions[:, i, :]  # (n_samples, k_dims)
            g_j = genre_positions[:, j, :]

            # Genre distance squared, normalized by tau
            genre_diff_sq = jnp.sum(((g_i - g_j) / tau) ** 2, axis=-1)  # (n_samples,)

            # Comparability probability
            p_comp = jax.nn.sigmoid(alpha - genre_diff_sq)

            # Posterior mean
            comparability = comparability.at[i, j].set(jnp.mean(p_comp))

    return comparability
