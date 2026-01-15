"""Tests for data parsing and models."""

import json
from pathlib import Path

import pytest

from src.data import (
    Artist,
    Comparison,
    ComparisonResult,
    Song,
    filter_artists_with_songs,
    filter_songs_by_artists,
    get_artist_pairs,
    get_compared_pairs,
    get_uncompared_pairs,
    parse_artists_json,
    parse_tracks_json,
    process_spotify_data,
)


@pytest.fixture
def sample_data():
    """Load sample Spotify data."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_spotify.json"
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture
def sample_artists(sample_data):
    """Parse artists from sample data."""
    return parse_artists_json(sample_data["artists"])


@pytest.fixture
def sample_tracks(sample_data):
    """Parse tracks from sample data."""
    return parse_tracks_json(sample_data["tracks"])


class TestParseArtists:
    def test_parse_artists_count(self, sample_artists):
        assert len(sample_artists) == 5

    def test_parse_artist_fields(self, sample_artists):
        radiohead = sample_artists[0]
        assert radiohead.id == "artist1"
        assert radiohead.name == "Radiohead"
        assert "alternative rock" in radiohead.genres
        assert radiohead.popularity == 82

    def test_parse_empty_json(self):
        artists = parse_artists_json({})
        assert artists == []

    def test_parse_missing_genres(self):
        data = {"items": [{"id": "x", "name": "Test Artist"}]}
        artists = parse_artists_json(data)
        assert len(artists) == 1
        assert artists[0].genres == []


class TestParseTracks:
    def test_parse_tracks_count(self, sample_tracks):
        # Should parse all 10 tracks
        assert len(sample_tracks) == 10

    def test_parse_track_fields(self, sample_tracks):
        track = sample_tracks[0]
        assert track.id == "track1"
        assert track.name == "Paranoid Android"
        assert track.artist_id == "artist1"
        assert track.artist_name == "Radiohead"

    def test_parse_empty_json(self):
        tracks = parse_tracks_json({})
        assert tracks == []


class TestFilterSongs:
    def test_filter_by_artists(self, sample_tracks):
        artist_ids = {"artist1", "artist2"}
        songs_by_artist = filter_songs_by_artists(sample_tracks, artist_ids)

        assert len(songs_by_artist["artist1"]) == 2
        assert len(songs_by_artist["artist2"]) == 2

    def test_filter_excludes_unknown(self, sample_tracks):
        artist_ids = {"artist1", "artist2", "artist3", "artist4", "artist5"}
        songs_by_artist = filter_songs_by_artists(sample_tracks, artist_ids)

        # track10 has unknown_artist, should not appear
        all_song_ids = []
        for songs in songs_by_artist.values():
            all_song_ids.extend(s.id for s in songs)
        assert "track10" not in all_song_ids


class TestFilterArtists:
    def test_filter_artists_with_songs(self, sample_artists, sample_tracks):
        artist_ids = {a.id for a in sample_artists}
        songs_by_artist = filter_songs_by_artists(sample_tracks, artist_ids)

        # Add an artist with no songs
        extra_artist = Artist(id="no_songs", name="No Songs Artist", genres=[])
        artists_with_extra = sample_artists + [extra_artist]
        songs_by_artist["no_songs"] = []

        filtered = filter_artists_with_songs(artists_with_extra, songs_by_artist)

        assert len(filtered) == 5  # Original 5 artists
        assert all(a.id != "no_songs" for a in filtered)


class TestProcessSpotifyData:
    def test_process_full_data(self, sample_data):
        artists, songs = process_spotify_data(
            sample_data["artists"],
            [sample_data["tracks"]],
            max_artists=25,
        )

        assert len(artists) == 5
        assert all(len(songs.get(a.id, [])) > 0 for a in artists)

    def test_max_artists_limit(self, sample_data):
        artists, songs = process_spotify_data(
            sample_data["artists"],
            [sample_data["tracks"]],
            max_artists=3,
        )

        assert len(artists) <= 3


class TestArtistPairs:
    def test_get_all_pairs(self, sample_artists):
        pairs = get_artist_pairs(sample_artists)
        # n*(n-1)/2 = 5*4/2 = 10 pairs
        assert len(pairs) == 10

    def test_pairs_are_unique(self, sample_artists):
        pairs = get_artist_pairs(sample_artists)
        # Normalize and check uniqueness
        normalized = [tuple(sorted(p)) for p in pairs]
        assert len(normalized) == len(set(normalized))


class TestComparedPairs:
    def test_get_compared_pairs(self, sample_artists):
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
                artist_a_id="artist3",
                artist_b_id="artist1",
                result=ComparisonResult.ABSTAIN,
            ),
        ]

        compared = get_compared_pairs(comparisons)

        assert ("artist1", "artist2") in compared or ("artist2", "artist1") in compared
        assert ("artist1", "artist3") in compared or ("artist3", "artist1") in compared

    def test_get_uncompared_pairs(self, sample_artists):
        comparisons = [
            Comparison(
                song_a_id="s1",
                song_b_id="s2",
                artist_a_id="artist1",
                artist_b_id="artist2",
                result=ComparisonResult.PREFER_A,
            ),
        ]

        uncompared = get_uncompared_pairs(sample_artists, comparisons)

        # Should have 10 - 1 = 9 pairs remaining
        assert len(uncompared) == 9

        # The compared pair should not be in uncompared
        for pair in uncompared:
            normalized = tuple(sorted(pair))
            assert normalized != ("artist1", "artist2")
