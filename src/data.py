"""Data models and Spotify JSON parsing for the artist ranking system."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Artist(BaseModel):
    """A Spotify artist."""

    id: str
    name: str
    genres: list[str] = Field(default_factory=list)
    popularity: int = 0
    image_url: str | None = None


class Song(BaseModel):
    """A Spotify track."""

    id: str
    name: str
    artist_id: str
    artist_name: str
    preview_url: str | None = None
    album_image_url: str | None = None
    album_name: str | None = None


class ComparisonResult(str, Enum):
    """Result of a pairwise comparison."""

    PREFER_A = "prefer_a"
    PREFER_B = "prefer_b"
    ABSTAIN = "abstain"


class Comparison(BaseModel):
    """A single pairwise comparison between two songs."""

    song_a_id: str
    song_b_id: str
    artist_a_id: str
    artist_b_id: str
    result: ComparisonResult


class SessionState(BaseModel):
    """Complete session state for persistence."""

    artists: list[Artist]
    songs: dict[str, list[Song]]  # artist_id -> songs
    comparisons: list[Comparison]


def parse_artists_json(data: dict[str, Any]) -> list[Artist]:
    """Parse Spotify 'Get User's Top Artists' response.

    Expected format:
    {
        "items": [
            {
                "id": "...",
                "name": "...",
                "genres": ["rock", "indie"],
                "popularity": 75
            },
            ...
        ]
    }
    """
    artists = []
    items = data.get("items", [])

    for item in items:
        # Get first (largest) image URL if available
        images = item.get("images", [])
        image_url = images[0]["url"] if images else None

        artist = Artist(
            id=item["id"],
            name=item["name"],
            genres=item.get("genres", []),
            popularity=item.get("popularity", 0),
            image_url=image_url,
        )
        artists.append(artist)

    return artists


def parse_tracks_json(data: dict[str, Any]) -> list[Song]:
    """Parse Spotify 'Get User's Top Tracks' response.

    Expected format:
    {
        "items": [
            {
                "id": "...",
                "name": "...",
                "artists": [{"id": "...", "name": "..."}],
                "preview_url": "..."
            },
            ...
        ]
    }
    """
    songs = []
    items = data.get("items", [])

    for item in items:
        artists = item.get("artists", [])
        if not artists:
            continue

        # Use primary (first) artist
        primary_artist = artists[0]

        # Get album art
        album = item.get("album", {})
        album_images = album.get("images", [])
        album_image_url = album_images[0]["url"] if album_images else None
        album_name = album.get("name")

        song = Song(
            id=item["id"],
            name=item["name"],
            artist_id=primary_artist["id"],
            artist_name=primary_artist["name"],
            preview_url=item.get("preview_url"),
            album_image_url=album_image_url,
            album_name=album_name,
        )
        songs.append(song)

    return songs


def filter_songs_by_artists(
    songs: list[Song], artist_ids: set[str]
) -> dict[str, list[Song]]:
    """Filter songs to only include those by the given artists.

    Returns a dict mapping artist_id to list of songs by that artist.
    """
    songs_by_artist: dict[str, list[Song]] = {aid: [] for aid in artist_ids}

    for song in songs:
        if song.artist_id in artist_ids:
            songs_by_artist[song.artist_id].append(song)

    return songs_by_artist


def filter_artists_with_songs(
    artists: list[Artist],
    songs_by_artist: dict[str, list[Song]],
    min_songs: int = 3,
) -> list[Artist]:
    """Remove artists that have fewer than min_songs after filtering."""
    return [a for a in artists if len(songs_by_artist.get(a.id, [])) >= min_songs]


def process_spotify_data(
    artists_json: dict[str, Any],
    tracks_json_list: list[dict[str, Any]],
    max_artists: int = 25,
    min_songs: int = 3,
) -> tuple[list[Artist], dict[str, list[Song]]]:
    """Process Spotify JSON data into artists and filtered songs.

    Args:
        artists_json: Response from Get User's Top Artists endpoint
        tracks_json_list: List of responses from Get User's Top Tracks endpoint
            (may be multiple due to pagination, 50 per page)
        max_artists: Maximum number of artists to consider
        min_songs: Minimum number of songs required to keep an artist

    Returns:
        Tuple of (filtered artists, songs grouped by artist)
    """
    # Parse artists (limit to top N)
    all_artists = parse_artists_json(artists_json)[:max_artists]
    artist_ids = {a.id for a in all_artists}

    # Parse all tracks from paginated responses
    all_songs: list[Song] = []
    for tracks_json in tracks_json_list:
        all_songs.extend(parse_tracks_json(tracks_json))

    # Filter songs to only those by our top artists
    songs_by_artist = filter_songs_by_artists(all_songs, artist_ids)

    # Remove artists with fewer than min_songs
    filtered_artists = filter_artists_with_songs(all_artists, songs_by_artist, min_songs)

    # Clean up songs dict to only include artists we're keeping
    final_artist_ids = {a.id for a in filtered_artists}
    songs_by_artist = {
        aid: songs for aid, songs in songs_by_artist.items() if aid in final_artist_ids
    }

    return filtered_artists, songs_by_artist


def get_artist_pairs(artists: list[Artist]) -> list[tuple[str, str]]:
    """Get all unique pairs of artist IDs."""
    pairs = []
    for i, a1 in enumerate(artists):
        for a2 in artists[i + 1 :]:
            pairs.append((a1.id, a2.id))
    return pairs


def get_compared_pairs(comparisons: list[Comparison]) -> set[tuple[str, str]]:
    """Get set of artist pairs that have been compared.

    Returns normalized pairs (smaller id first) for consistent lookup.
    """
    pairs = set()
    for c in comparisons:
        pair = tuple(sorted([c.artist_a_id, c.artist_b_id]))
        pairs.add(pair)
    return pairs


def get_uncompared_pairs(
    artists: list[Artist], comparisons: list[Comparison]
) -> list[tuple[str, str]]:
    """Get artist pairs that haven't been compared yet."""
    all_pairs = get_artist_pairs(artists)
    compared = get_compared_pairs(comparisons)

    uncompared = []
    for pair in all_pairs:
        normalized = tuple(sorted(pair))
        if normalized not in compared:
            uncompared.append(pair)

    return uncompared


def load_from_data_dir(
    data_dir: str | Path = "data",
    max_artists: int = 25,
    min_songs: int = 3,
) -> tuple[list[Artist], dict[str, list[Song]]]:
    """Load Spotify data from the data directory.

    Expects:
    - data/top_artists.json
    - data/top_tracks_*.json (one or more paginated files)

    Args:
        data_dir: Path to data directory
        max_artists: Maximum number of artists to include
        min_songs: Minimum number of songs required to keep an artist

    Returns:
        Tuple of (filtered artists, songs grouped by artist)
    """
    data_path = Path(data_dir)

    # Load artists
    artists_file = data_path / "top_artists.json"
    if not artists_file.exists():
        raise FileNotFoundError(f"Artists file not found: {artists_file}")

    with open(artists_file) as f:
        artists_json = json.load(f)

    # Load all tracks files (handle pagination)
    tracks_json_list = []
    for tracks_file in sorted(data_path.glob("top_tracks*.json")):
        with open(tracks_file) as f:
            tracks_json_list.append(json.load(f))

    if not tracks_json_list:
        raise FileNotFoundError(f"No tracks files found in {data_path}")

    return process_spotify_data(artists_json, tracks_json_list, max_artists=max_artists, min_songs=min_songs)
