"""Tests for UI components (non-Streamlit parts)."""

import pytest

from src.data import Artist, Comparison, ComparisonResult, Song
from src.model import compute_ranking_confidence


class TestConfidenceDisplay:
    """Test confidence computation for UI display."""

    def test_zero_comparisons(self):
        """Test display with no comparisons."""
        confidence = compute_ranking_confidence(
            n_comparisons=0,
            total_pairs=100,
        )

        assert confidence["percent_complete"] == 0.0
        assert confidence["status"] == "warming_up"
        assert confidence["show_accuracy"] is False

    def test_below_minimum(self):
        """Test display below minimum threshold."""
        confidence = compute_ranking_confidence(
            n_comparisons=5,
            total_pairs=100,
        )

        assert confidence["status"] == "warming_up"
        assert confidence["comparisons_to_next"] == 5  # Need 5 more to reach 10

    def test_early_ranking(self):
        """Test early ranking phase (10-25%)."""
        confidence = compute_ranking_confidence(
            n_comparisons=15,
            total_pairs=100,
        )

        assert confidence["status"] == "early"
        assert confidence["show_accuracy"] is False

    def test_refining_phase(self):
        """Test refining phase (25-50%)."""
        confidence = compute_ranking_confidence(
            n_comparisons=30,
            total_pairs=100,
        )

        assert confidence["status"] == "refining"
        assert confidence["show_accuracy"] is True

    def test_solid_ranking(self):
        """Test solid ranking phase (50%+)."""
        confidence = compute_ranking_confidence(
            n_comparisons=60,
            total_pairs=100,
        )

        assert confidence["status"] == "solid"
        assert confidence["show_accuracy"] is True

    def test_complete(self):
        """Test 100% completion."""
        confidence = compute_ranking_confidence(
            n_comparisons=100,
            total_pairs=100,
        )

        assert confidence["percent_complete"] == 100.0
        assert confidence["status"] == "solid"


class TestSongDisplay:
    """Test Song model for display."""

    def test_song_with_album_art(self):
        """Test song model includes album art."""
        song = Song(
            id="test123",
            name="Test Song",
            artist_id="artist1",
            artist_name="Test Artist",
            preview_url=None,
            album_image_url="https://example.com/album.jpg",
            album_name="Test Album",
        )

        assert song.album_image_url == "https://example.com/album.jpg"
        assert song.album_name == "Test Album"

    def test_song_without_album_art(self):
        """Test song model without album art."""
        song = Song(
            id="test123",
            name="Test Song",
            artist_id="artist1",
            artist_name="Test Artist",
        )

        assert song.album_image_url is None
        assert song.album_name is None


class TestArtistDisplay:
    """Test Artist model for display."""

    def test_artist_with_image(self):
        """Test artist model includes image."""
        artist = Artist(
            id="artist1",
            name="Test Artist",
            genres=["rock", "indie"],
            popularity=75,
            image_url="https://example.com/artist.jpg",
        )

        assert artist.image_url == "https://example.com/artist.jpg"

    def test_artist_genres(self):
        """Test artist genres are preserved."""
        artist = Artist(
            id="artist1",
            name="Test Artist",
            genres=["alternative rock", "art rock", "experimental"],
        )

        assert len(artist.genres) == 3
        assert "alternative rock" in artist.genres
