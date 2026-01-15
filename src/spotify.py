"""Spotify OAuth and API client for fetching user's top artists and tracks."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

try:
    import spotipy
    from spotipy.cache_handler import MemoryCacheHandler
    from spotipy.oauth2 import SpotifyOAuth

    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False


def get_spotify_credentials() -> tuple[str | None, str | None, str | None]:
    """Get Spotify credentials from secrets or environment variables.

    Returns:
        Tuple of (client_id, client_secret, redirect_uri)
    """
    # Try Streamlit secrets first
    try:
        client_id = st.secrets.get("SPOTIPY_CLIENT_ID")
        client_secret = st.secrets.get("SPOTIPY_CLIENT_SECRET")
        redirect_uri = st.secrets.get("SPOTIPY_REDIRECT_URI")
        if client_id and client_secret:
            return client_id, client_secret, redirect_uri
    except Exception:
        pass

    # Fall back to environment variables
    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.environ.get("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8501")

    return client_id, client_secret, redirect_uri


def has_spotify_credentials() -> bool:
    """Check if Spotify credentials are configured."""
    if not SPOTIPY_AVAILABLE:
        return False
    client_id, client_secret, _ = get_spotify_credentials()
    return bool(client_id and client_secret)


def create_auth_manager() -> SpotifyOAuth | None:
    """Create a SpotifyOAuth manager with memory-based caching.

    Returns:
        SpotifyOAuth instance or None if credentials are missing
    """
    if not SPOTIPY_AVAILABLE:
        return None

    client_id, client_secret, redirect_uri = get_spotify_credentials()

    if not client_id or not client_secret:
        return None

    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri or "http://127.0.0.1:8501",
        scope="user-top-read",
        cache_handler=MemoryCacheHandler(),
        show_dialog=True,
    )


def get_auth_url(auth_manager: SpotifyOAuth) -> str:
    """Get the Spotify authorization URL.

    Args:
        auth_manager: SpotifyOAuth instance

    Returns:
        Authorization URL to redirect user to
    """
    return auth_manager.get_authorize_url()


def exchange_code_for_token(auth_manager: SpotifyOAuth, code: str) -> dict | None:
    """Exchange authorization code for access token.

    Args:
        auth_manager: SpotifyOAuth instance
        code: Authorization code from Spotify callback

    Returns:
        Token info dict or None on failure
    """
    try:
        token_info = auth_manager.get_access_token(code, as_dict=True, check_cache=False)
        return token_info
    except Exception:
        return None


def get_spotify_client(auth_manager: SpotifyOAuth) -> spotipy.Spotify | None:
    """Create a Spotify client from an authenticated auth manager.

    Args:
        auth_manager: SpotifyOAuth instance with valid token

    Returns:
        Spotify client or None on failure
    """
    if not SPOTIPY_AVAILABLE:
        return None

    try:
        return spotipy.Spotify(auth_manager=auth_manager)
    except Exception:
        return None


def fetch_top_artists(
    sp: spotipy.Spotify,
    limit: int = 25,
    time_range: str = "medium_term",
) -> dict[str, Any]:
    """Fetch user's top artists.

    Args:
        sp: Authenticated Spotify client
        limit: Number of artists to fetch (max 50)
        time_range: Time range - "short_term" (4 weeks), "medium_term" (6 months), "long_term" (1 year)

    Returns:
        API response with items array
    """
    return sp.current_user_top_artists(limit=min(limit, 50), time_range=time_range)


def fetch_top_tracks(
    sp: spotipy.Spotify,
    limit: int = 250,
    time_range: str = "medium_term",
) -> dict[str, Any]:
    """Fetch user's top tracks with pagination.

    Args:
        sp: Authenticated Spotify client
        limit: Total number of tracks to fetch
        time_range: Time range - "short_term", "medium_term", "long_term"

    Returns:
        Combined response with all items
    """
    all_items = []
    offset = 0
    per_page = 50

    while offset < limit:
        batch_limit = min(per_page, limit - offset)
        result = sp.current_user_top_tracks(
            limit=batch_limit,
            offset=offset,
            time_range=time_range,
        )

        items = result.get("items", [])
        all_items.extend(items)

        # Stop if no more items
        if not result.get("next") or len(items) < batch_limit:
            break

        offset += per_page

    return {"items": all_items}


def fetch_spotify_data(
    sp: spotipy.Spotify,
    max_artists: int = 25,
    max_tracks: int = 250,
    time_range: str = "medium_term",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch both top artists and top tracks.

    Args:
        sp: Authenticated Spotify client
        max_artists: Maximum number of artists to fetch
        max_tracks: Maximum number of tracks to fetch
        time_range: Time range for both queries

    Returns:
        Tuple of (artists_json, tracks_json) in Spotify API format
    """
    artists_json = fetch_top_artists(sp, limit=max_artists, time_range=time_range)
    tracks_json = fetch_top_tracks(sp, limit=max_tracks, time_range=time_range)

    return artists_json, tracks_json
