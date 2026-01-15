"""Shared fixtures for all tests."""

import json
from pathlib import Path

import pytest

from src.data import (
    Artist,
    Comparison,
    ComparisonResult,
    Song,
    parse_artists_json,
    parse_tracks_json,
    filter_songs_by_artists,
)


@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_spotify_data(fixtures_dir):
    """Load sample Spotify data from fixtures."""
    with open(fixtures_dir / "sample_spotify.json") as f:
        return json.load(f)


@pytest.fixture
def sample_artists(sample_spotify_data):
    """Parse artists from sample data."""
    return parse_artists_json(sample_spotify_data["artists"])


@pytest.fixture
def sample_songs(sample_spotify_data, sample_artists):
    """Parse and filter songs from sample data."""
    all_songs = parse_tracks_json(sample_spotify_data["tracks"])
    artist_ids = {a.id for a in sample_artists}
    return filter_songs_by_artists(all_songs, artist_ids)


@pytest.fixture
def artist_id_to_idx(sample_artists):
    """Create mapping from artist ID to index."""
    return {a.id: i for i, a in enumerate(sample_artists)}


@pytest.fixture
def sample_comparisons():
    """Create a set of sample comparisons for testing."""
    return [
        Comparison(
            song_a_id="track1",
            song_b_id="track3",
            artist_a_id="artist1",
            artist_b_id="artist2",
            result=ComparisonResult.PREFER_A,
        ),
        Comparison(
            song_a_id="track3",
            song_b_id="track5",
            artist_a_id="artist2",
            artist_b_id="artist3",
            result=ComparisonResult.PREFER_B,
        ),
        Comparison(
            song_a_id="track1",
            song_b_id="track7",
            artist_a_id="artist1",
            artist_b_id="artist4",
            result=ComparisonResult.ABSTAIN,
        ),
        Comparison(
            song_a_id="track5",
            song_b_id="track8",
            artist_a_id="artist3",
            artist_b_id="artist5",
            result=ComparisonResult.PREFER_A,
        ),
        Comparison(
            song_a_id="track2",
            song_b_id="track4",
            artist_a_id="artist1",
            artist_b_id="artist2",
            result=ComparisonResult.PREFER_B,
        ),
    ]


@pytest.fixture
def many_comparisons(sample_artists):
    """Create enough comparisons to trigger inference (10+)."""
    comparisons = []
    artist_list = list(sample_artists)

    for i in range(15):
        a1 = artist_list[i % len(artist_list)]
        a2 = artist_list[(i + 1) % len(artist_list)]

        # Vary the results
        if i % 3 == 0:
            result = ComparisonResult.PREFER_A
        elif i % 3 == 1:
            result = ComparisonResult.PREFER_B
        else:
            result = ComparisonResult.ABSTAIN

        comp = Comparison(
            song_a_id=f"song_{i}_a",
            song_b_id=f"song_{i}_b",
            artist_a_id=a1.id,
            artist_b_id=a2.id,
            result=result,
        )
        comparisons.append(comp)

    return comparisons
