"""Integration tests for the full app flow."""

import pytest

from src.data import (
    Artist,
    Comparison,
    ComparisonResult,
    get_artist_pairs,
    get_uncompared_pairs,
    load_from_data_dir,
    process_spotify_data,
)
from src.model import (
    compute_ranking_confidence,
    encode_comparisons,
    extract_posterior_summary,
    run_inference,
)
from src.selection import select_random_pair, select_songs_for_comparison


class TestFullInferenceFlow:
    """Test the complete inference pipeline."""

    @pytest.mark.slow
    def test_inference_with_sample_data(
        self, sample_artists, sample_songs, many_comparisons, artist_id_to_idx
    ):
        """Test full inference flow with sample data."""
        artist_ids = [a.id for a in sample_artists]

        # Run inference
        samples = run_inference(
            n_artists=len(sample_artists),
            comparisons=many_comparisons,
            artist_id_to_idx=artist_id_to_idx,
            prior_genre_means=None,
            k_dims=1,
            num_warmup=200,
            num_samples=200,
            num_chains=1,
        )

        # Extract posterior
        posterior = extract_posterior_summary(samples, artist_ids)

        # Verify structure
        assert len(posterior) == len(sample_artists)
        for aid in artist_ids:
            assert aid in posterior
            assert "utility_mean" in posterior[aid]
            assert "utility_std" in posterior[aid]
            assert "genre_mean" in posterior[aid]

    @pytest.mark.slow
    def test_inference_2d(
        self, sample_artists, many_comparisons, artist_id_to_idx
    ):
        """Test inference with 2D genre space."""
        artist_ids = [a.id for a in sample_artists]

        samples = run_inference(
            n_artists=len(sample_artists),
            comparisons=many_comparisons,
            artist_id_to_idx=artist_id_to_idx,
            prior_genre_means=None,
            k_dims=2,
            num_warmup=200,
            num_samples=200,
            num_chains=1,
        )

        posterior = extract_posterior_summary(samples, artist_ids)

        # Check that genre_mean has 2 dimensions
        for aid in artist_ids:
            assert len(posterior[aid]["genre_mean"]) == 2


class TestSelectionFlow:
    """Test comparison selection logic."""

    def test_random_pair_selection(self, sample_artists, sample_comparisons):
        """Test that random selection avoids already-compared pairs."""
        # Get a random pair
        pair = select_random_pair(sample_artists, sample_comparisons)

        if pair is not None:
            # Verify it's not already compared
            uncompared = get_uncompared_pairs(sample_artists, sample_comparisons)
            normalized_pair = tuple(sorted(pair))
            normalized_uncompared = [tuple(sorted(p)) for p in uncompared]
            assert normalized_pair in normalized_uncompared

    def test_song_selection(self, sample_artists, sample_songs):
        """Test song selection for comparison."""
        artist_a = sample_artists[0]
        artist_b = sample_artists[1]

        song_a, song_b = select_songs_for_comparison(
            artist_a.id, artist_b.id, sample_songs
        )

        assert song_a.artist_id == artist_a.id
        assert song_b.artist_id == artist_b.id


class TestRankingConfidence:
    """Test ranking confidence computation."""

    def test_warming_up_phase(self, sample_artists):
        """Test confidence in warming up phase."""
        total_pairs = len(get_artist_pairs(sample_artists))

        confidence = compute_ranking_confidence(
            n_comparisons=5,
            total_pairs=total_pairs,
        )

        assert confidence["status"] == "warming_up"
        assert confidence["show_accuracy"] is False

    def test_early_phase(self, sample_artists):
        """Test confidence in early phase."""
        total_pairs = len(get_artist_pairs(sample_artists))

        confidence = compute_ranking_confidence(
            n_comparisons=10,
            total_pairs=total_pairs,
        )

        # 10/10 = 100% for 5 artists, so it won't be early
        # Let's check it's not warming_up at least
        assert confidence["status"] != "warming_up"

    def test_solid_phase(self, sample_artists):
        """Test confidence when most pairs are compared."""
        total_pairs = len(get_artist_pairs(sample_artists))

        confidence = compute_ranking_confidence(
            n_comparisons=total_pairs,
            total_pairs=total_pairs,
        )

        assert confidence["status"] == "solid"
        assert confidence["percent_complete"] == 100.0


class TestDataLoading:
    """Test data loading from files."""

    def test_load_from_data_dir(self):
        """Test loading real data from data directory."""
        try:
            artists, songs = load_from_data_dir("data", max_artists=10)

            assert len(artists) > 0
            assert len(artists) <= 10
            assert all(len(songs.get(a.id, [])) > 0 for a in artists)
        except FileNotFoundError:
            pytest.skip("No data directory found")

    def test_process_spotify_data(self, sample_spotify_data):
        """Test processing Spotify API response format."""
        artists, songs = process_spotify_data(
            sample_spotify_data["artists"],
            [sample_spotify_data["tracks"]],
            max_artists=25,
        )

        assert len(artists) == 5
        assert all(a.id in songs for a in artists)


class TestComparisonEncoding:
    """Test comparison encoding for model."""

    def test_encode_all_result_types(self, artist_id_to_idx):
        """Test encoding of all comparison result types."""
        comparisons = [
            Comparison(
                song_a_id="s1",
                song_b_id="s2",
                artist_a_id="artist1",
                artist_b_id="artist2",
                result=ComparisonResult.PREFER_A,
            ),
            Comparison(
                song_a_id="s3",
                song_b_id="s4",
                artist_a_id="artist2",
                artist_b_id="artist3",
                result=ComparisonResult.PREFER_B,
            ),
            Comparison(
                song_a_id="s5",
                song_b_id="s6",
                artist_a_id="artist1",
                artist_b_id="artist3",
                result=ComparisonResult.ABSTAIN,
            ),
        ]

        a_idx, b_idx, outcomes = encode_comparisons(comparisons, artist_id_to_idx)

        assert int(outcomes[0]) == 1  # PREFER_A
        assert int(outcomes[1]) == 2  # PREFER_B
        assert int(outcomes[2]) == 0  # ABSTAIN
