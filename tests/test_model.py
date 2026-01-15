"""Tests for the NumPyro model."""

import jax.numpy as jnp
import numpy as np
import pytest

from src.data import Artist, Comparison, ComparisonResult
from src.model import (
    artist_preference_model,
    compute_comparability_matrix,
    encode_comparisons,
    extract_posterior_summary,
    run_inference,
)


@pytest.fixture
def sample_artists():
    """Create sample artists for testing."""
    return [
        Artist(id="a1", name="Artist 1", genres=["rock"]),
        Artist(id="a2", name="Artist 2", genres=["pop"]),
        Artist(id="a3", name="Artist 3", genres=["jazz"]),
    ]


@pytest.fixture
def sample_comparisons():
    """Create sample comparisons."""
    return [
        Comparison(
            song_a_id="s1",
            song_b_id="s2",
            artist_a_id="a1",
            artist_b_id="a2",
            result=ComparisonResult.PREFER_A,
        ),
        Comparison(
            song_a_id="s3",
            song_b_id="s4",
            artist_a_id="a2",
            artist_b_id="a3",
            result=ComparisonResult.PREFER_B,
        ),
        Comparison(
            song_a_id="s5",
            song_b_id="s6",
            artist_a_id="a1",
            artist_b_id="a3",
            result=ComparisonResult.ABSTAIN,
        ),
    ]


@pytest.fixture
def artist_id_to_idx(sample_artists):
    """Create ID to index mapping."""
    return {a.id: i for i, a in enumerate(sample_artists)}


class TestEncodeComparisons:
    def test_encode_basic(self, sample_comparisons, artist_id_to_idx):
        a_idx, b_idx, outcomes = encode_comparisons(sample_comparisons, artist_id_to_idx)

        assert len(a_idx) == 3
        assert len(b_idx) == 3
        assert len(outcomes) == 3

        # Check outcomes encoding
        assert outcomes[0] == 1  # PREFER_A
        assert outcomes[1] == 2  # PREFER_B
        assert outcomes[2] == 0  # ABSTAIN

    def test_encode_indices(self, sample_comparisons, artist_id_to_idx):
        a_idx, b_idx, _ = encode_comparisons(sample_comparisons, artist_id_to_idx)

        # First comparison: a1 vs a2
        assert a_idx[0] == 0
        assert b_idx[0] == 1

        # Second comparison: a2 vs a3
        assert a_idx[1] == 1
        assert b_idx[1] == 2


class TestInference:
    @pytest.mark.slow
    def test_inference_runs(self, sample_artists, sample_comparisons, artist_id_to_idx):
        """Test that inference completes without error."""
        samples = run_inference(
            n_artists=len(sample_artists),
            comparisons=sample_comparisons,
            artist_id_to_idx=artist_id_to_idx,
            prior_genre_means=None,
            k_dims=1,
            num_warmup=100,
            num_samples=100,
            num_chains=1,
            seed=42,
        )

        # Check that we got samples
        assert "utilities" in samples
        assert "genre_positions" in samples
        assert "lambdas" in samples

        # Check shapes
        assert samples["utilities"].shape == (100, 3)
        assert samples["genre_positions"].shape == (100, 3, 1)
        assert samples["lambdas"].shape == (100, 3)

    @pytest.mark.slow
    def test_inference_with_priors(
        self, sample_artists, sample_comparisons, artist_id_to_idx
    ):
        """Test inference with LLM-derived priors."""
        prior_means = jnp.array([[0.0], [1.0], [-1.0]])

        samples = run_inference(
            n_artists=len(sample_artists),
            comparisons=sample_comparisons,
            artist_id_to_idx=artist_id_to_idx,
            prior_genre_means=prior_means,
            k_dims=1,
            num_warmup=100,
            num_samples=100,
            num_chains=1,
            seed=42,
        )

        assert "genre_positions" in samples


class TestPosteriorSummary:
    def test_extract_summary(self, sample_artists):
        """Test posterior summary extraction."""
        # Create mock samples
        n_samples = 100
        n_artists = 3

        samples = {
            "utilities": jnp.array(
                np.random.randn(n_samples, n_artists).astype(np.float32)
            ),
            "genre_positions": jnp.array(
                np.random.randn(n_samples, n_artists, 1).astype(np.float32)
            ),
            "lambdas": jnp.array(
                np.abs(np.random.randn(n_samples, n_artists)).astype(np.float32) + 0.5
            ),
        }

        artist_ids = [a.id for a in sample_artists]
        summary = extract_posterior_summary(samples, artist_ids)

        assert len(summary) == 3
        assert "a1" in summary

        # Check structure
        for aid in artist_ids:
            assert "utility_mean" in summary[aid]
            assert "utility_std" in summary[aid]
            assert "genre_mean" in summary[aid]
            assert "genre_std" in summary[aid]
            assert "lambda_mean" in summary[aid]


class TestComparabilityMatrix:
    def test_compute_comparability(self, sample_artists):
        """Test comparability matrix computation."""
        n_samples = 50
        n_artists = 3

        # Create mock samples with distinct genre positions
        genre_positions = np.zeros((n_samples, n_artists, 1))
        genre_positions[:, 0, 0] = 0.0  # Artist 1 at origin
        genre_positions[:, 1, 0] = 0.5  # Artist 2 nearby
        genre_positions[:, 2, 0] = 3.0  # Artist 3 far away

        samples = {
            "genre_positions": jnp.array(genre_positions.astype(np.float32)),
            "alpha": jnp.ones(n_samples),
            "tau": jnp.ones((n_samples, 1)),
        }

        matrix = compute_comparability_matrix(samples, n_artists)

        # Check shape
        assert matrix.shape == (3, 3)

        # Diagonal should be 0 (we skip i==j)
        for i in range(n_artists):
            assert matrix[i, i] == 0.0

        # Artists 1 and 2 should be more comparable than 1 and 3
        assert float(matrix[0, 1]) > float(matrix[0, 2])
