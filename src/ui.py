"""Streamlit UI components for the artist ranking app."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.data import (
    Artist,
    Comparison,
    ComparisonResult,
    Song,
    load_from_data_dir,
    process_spotify_data,
)
from src.priors import get_default_priors, get_genre_priors_from_llm_sync


def _process_and_store_data(
    artists_json: dict,
    tracks_json_list: list[dict],
    max_artists: int,
    k_dims: int,
    use_llm_priors: bool,
) -> bool:
    """Process Spotify data and store in session state.

    Returns:
        True if successful
    """
    try:
        artists, songs_by_artist = process_spotify_data(
            artists_json, tracks_json_list, max_artists=max_artists
        )

        if not artists:
            st.error("No valid artists found after processing.")
            return False

        # Generate genre priors
        prior_genre_means = None
        if use_llm_priors:
            with st.spinner("Generating genre priors from LLM..."):
                try:
                    prior_genre_means = get_genre_priors_from_llm_sync(
                        artists, k_dims=k_dims
                    )
                    st.success("Generated genre priors!")
                except Exception as e:
                    st.warning(f"Failed to generate LLM priors: {e}. Using defaults.")
                    prior_genre_means = get_default_priors(len(artists), k_dims=k_dims)
        else:
            prior_genre_means = get_default_priors(len(artists), k_dims=k_dims)

        # Store in session state
        st.session_state.artists = artists
        st.session_state.songs = songs_by_artist
        st.session_state.comparisons = []
        st.session_state.posterior = None
        st.session_state.prior_genre_means = prior_genre_means
        st.session_state.k_dims = k_dims
        st.session_state.setup_complete = True

        st.success(f"Loaded {len(artists)} artists with songs!")
        return True

    except Exception as e:
        st.error(f"Error processing data: {e}")
        return False


def render_setup_page() -> bool:
    """Render the setup page for loading Spotify data.

    Returns:
        True if setup is complete and we can proceed to comparisons
    """
    # Check for Spotify OAuth credentials
    from src.spotify import has_spotify_credentials, create_auth_manager, get_auth_url

    has_spotify_creds = has_spotify_credentials()

    # Check if we have Spotify data from OAuth callback
    spotify_connected = st.session_state.get("spotify_connected", False)
    has_spotify_data = "spotify_artists_json" in st.session_state

    # Check if data directory exists with files
    data_dir = Path("data")
    has_data_files = (
        data_dir.exists()
        and (data_dir / "top_artists.json").exists()
        and list(data_dir.glob("top_tracks*.json"))
    )

    # === Quick Start Section ===
    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # If user has already connected to Spotify
    if spotify_connected and has_spotify_data:
        _render_spotify_config_section()
        return st.session_state.get("setup_complete", False)

    # If data files exist, offer quick start
    if has_data_files:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="
                background: linear-gradient(145deg, #1DB954, #169c46);
                border-radius: 12px;
                padding: 24px;
                text-align: center;
                margin-bottom: 24px;
            ">
                <p style="color: #000; font-weight: 600; margin: 0;">
                    Found your Spotify data in the data/ folder
                </p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Start Ranking", type="primary", use_container_width=True, key="quick_start"):
                # Use smart defaults: 25 artists, 2D, LLM priors
                _load_from_data_dir_with_defaults(data_dir)

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        # Settings expander
        with st.expander("Customize Settings"):
            _render_settings_panel(data_dir, has_data_files)

    else:
        # No data files - show data loading options
        _render_data_loading_options(has_spotify_creds, data_dir)

    return st.session_state.get("setup_complete", False)


def _load_from_data_dir_with_defaults(data_dir: Path) -> None:
    """Load data from directory with smart defaults."""
    try:
        artists, songs_by_artist = load_from_data_dir(data_dir, max_artists=25)

        if not artists:
            st.error("No valid artists found after processing.")
            return

        # Smart defaults: 2D with LLM priors
        k_dims = 2
        use_llm_priors = True

        prior_genre_means = None
        if use_llm_priors:
            with st.spinner("Generating genre priors from LLM..."):
                try:
                    prior_genre_means = get_genre_priors_from_llm_sync(artists, k_dims=k_dims)
                except Exception as e:
                    st.warning(f"Using default priors: {e}")
                    prior_genre_means = get_default_priors(len(artists), k_dims=k_dims)
        else:
            prior_genre_means = get_default_priors(len(artists), k_dims=k_dims)

        # Store in session state
        st.session_state.artists = artists
        st.session_state.songs = songs_by_artist
        st.session_state.comparisons = []
        st.session_state.posterior = None
        st.session_state.prior_genre_means = prior_genre_means
        st.session_state.k_dims = k_dims
        st.session_state.setup_complete = True

        st.rerun()

    except Exception as e:
        st.error(f"Error loading data: {e}")


def _render_settings_panel(data_dir: Path, has_data_files: bool) -> None:
    """Render the settings customization panel."""
    col1, col2 = st.columns(2)

    with col1:
        max_artists = st.slider(
            "Number of artists",
            min_value=5,
            max_value=50,
            value=25,
            help="More artists = more comparisons needed",
            key="settings_max_artists",
        )

    with col2:
        k_dims = st.radio(
            "Genre dimensions",
            options=[1, 2],
            index=1,  # Default to 2D
            horizontal=True,
            help="2D gives richer visualization",
            key="settings_k_dims",
        )

    use_llm_priors = st.checkbox(
        "Use AI to initialize genre positions",
        value=True,
        help="Uses Claude to estimate initial genre distances (recommended)",
        key="settings_llm_priors",
    )

    if has_data_files:
        if st.button("Start with Custom Settings", key="custom_start"):
            try:
                artists, songs_by_artist = load_from_data_dir(data_dir, max_artists=max_artists)

                if not artists:
                    st.error("No valid artists found.")
                    return

                prior_genre_means = None
                if use_llm_priors:
                    with st.spinner("Generating genre priors..."):
                        try:
                            prior_genre_means = get_genre_priors_from_llm_sync(artists, k_dims=k_dims)
                        except Exception as e:
                            st.warning(f"Using defaults: {e}")
                            prior_genre_means = get_default_priors(len(artists), k_dims=k_dims)
                else:
                    prior_genre_means = get_default_priors(len(artists), k_dims=k_dims)

                st.session_state.artists = artists
                st.session_state.songs = songs_by_artist
                st.session_state.comparisons = []
                st.session_state.posterior = None
                st.session_state.prior_genre_means = prior_genre_means
                st.session_state.k_dims = k_dims
                st.session_state.setup_complete = True

                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")


def _render_spotify_config_section() -> None:
    """Render configuration after Spotify OAuth success."""
    st.success("Connected to Spotify!")

    artists_json = st.session_state.spotify_artists_json
    tracks_json = st.session_state.spotify_tracks_json

    n_artists = len(artists_json.get("items", []))
    n_tracks = len(tracks_json.get("items", []))

    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <p style="font-size: 1.1rem;">
            Found <strong>{n_artists} artists</strong> and <strong>{n_tracks} tracks</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Ranking", type="primary", use_container_width=True):
            # Smart defaults
            if _process_and_store_data(
                artists_json,
                [tracks_json],
                max_artists=min(25, n_artists),
                k_dims=2,
                use_llm_priors=True,
            ):
                st.rerun()

    with st.expander("Customize Settings"):
        max_artists = st.slider(
            "Number of artists",
            min_value=5,
            max_value=min(50, n_artists),
            value=min(25, n_artists),
            key="spotify_max_artists",
        )

        k_dims = st.radio(
            "Genre dimensions",
            options=[1, 2],
            index=1,
            horizontal=True,
            key="spotify_k_dims",
        )

        use_llm_priors = st.checkbox(
            "Use AI to initialize genre positions",
            value=True,
            key="spotify_llm_priors",
        )

        if st.button("Start with Custom Settings", key="spotify_custom_start"):
            if _process_and_store_data(
                artists_json,
                [tracks_json],
                max_artists,
                k_dims,
                use_llm_priors,
            ):
                st.rerun()


def _fetch_spotify_data_with_token(token: str) -> None:
    """Fetch Spotify data using a user-provided access token."""
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    base_url = "https://api.spotify.com/v1/me/top"

    try:
        with st.spinner("Fetching your top artists..."):
            artists_res = requests.get(
                f"{base_url}/artists",
                headers=headers,
                params={"limit": 50, "time_range": "long_term"},
                timeout=30,
            )

            if artists_res.status_code == 401:
                st.error("Token expired or invalid. Please get a fresh token.")
                return
            elif artists_res.status_code == 403:
                st.error("Token doesn't have the right permissions. Make sure you authorized with user-top-read scope.")
                return
            elif not artists_res.ok:
                st.error(f"Failed to fetch artists: {artists_res.status_code}")
                return

            artists_json = artists_res.json()

        with st.spinner("Fetching your top tracks..."):
            all_tracks = []
            for offset in range(0, 250, 50):
                tracks_res = requests.get(
                    f"{base_url}/tracks",
                    headers=headers,
                    params={"limit": 50, "offset": offset, "time_range": "long_term"},
                    timeout=30,
                )
                if not tracks_res.ok:
                    break
                batch = tracks_res.json()
                all_tracks.extend(batch.get("items", []))
                if not batch.get("next"):
                    break

            tracks_json = {"items": all_tracks}

        # Process the data
        artists, songs_by_artist = process_spotify_data(
            artists_json, [tracks_json], max_artists=25
        )

        if not artists:
            st.error("No valid artists found. Try a different time range on the Spotify page.")
            return

        k_dims = 2
        prior_genre_means = None

        with st.spinner("Setting up your ranking..."):
            try:
                prior_genre_means = get_genre_priors_from_llm_sync(artists, k_dims=k_dims)
            except Exception:
                prior_genre_means = get_default_priors(len(artists), k_dims=k_dims)

        st.session_state.artists = artists
        st.session_state.songs = songs_by_artist
        st.session_state.comparisons = []
        st.session_state.posterior = None
        st.session_state.prior_genre_means = prior_genre_means
        st.session_state.k_dims = k_dims
        st.session_state.setup_complete = True

        st.success(f"Loaded {len(artists)} artists with {sum(len(s) for s in songs_by_artist.values())} tracks!")
        st.rerun()

    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")


def _process_uploaded_file(uploaded_file) -> None:
    """Process an uploaded JSON file with Spotify data."""
    try:
        data = json.loads(uploaded_file.read().decode("utf-8"))

        # Handle different formats
        if "artists" in data and "tracks" in data:
            # Bookmarklet/export format
            artists_json = data["artists"]
            tracks_json_list = [data["tracks"]]
        elif "items" in data:
            st.error("This looks like a single endpoint response. Please upload a combined export file.")
            return
        else:
            st.error("Unrecognized file format.")
            return

        artists, songs_by_artist = process_spotify_data(
            artists_json, tracks_json_list, max_artists=25
        )

        if not artists:
            st.error("No valid artists found in the file.")
            return

        k_dims = 2
        prior_genre_means = None

        with st.spinner("Setting up your ranking..."):
            try:
                prior_genre_means = get_genre_priors_from_llm_sync(artists, k_dims=k_dims)
            except Exception:
                prior_genre_means = get_default_priors(len(artists), k_dims=k_dims)

        st.session_state.artists = artists
        st.session_state.songs = songs_by_artist
        st.session_state.comparisons = []
        st.session_state.posterior = None
        st.session_state.prior_genre_means = prior_genre_means
        st.session_state.k_dims = k_dims
        st.session_state.setup_complete = True

        st.success(f"Loaded {len(artists)} artists!")
        st.rerun()

    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON file: {e}")
    except Exception as e:
        st.error(f"Error processing file: {e}")


def _render_data_loading_options(has_spotify_creds: bool, data_dir: Path) -> None:
    """Render data loading options when no data is available."""
    # Center the options
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Option 1: Spotify OAuth (if available)
        if has_spotify_creds:
            from src.spotify import create_auth_manager, get_auth_url

            st.markdown("""
            <div style="
                background: linear-gradient(145deg, #1DB954, #169c46);
                border-radius: 12px;
                padding: 24px;
                text-align: center;
                margin-bottom: 16px;
            ">
                <p style="color: #000; font-weight: 600; margin: 0;">
                    Connect directly to Spotify (easiest)
                </p>
            </div>
            """, unsafe_allow_html=True)

            time_range = st.radio(
                "Time range:",
                options=["short_term", "medium_term", "long_term"],
                format_func=lambda x: {
                    "short_term": "Last 4 weeks",
                    "medium_term": "Last 6 months",
                    "long_term": "All time",
                }[x],
                index=1,
                horizontal=True,
                key="time_range",
            )
            st.session_state.spotify_time_range = time_range

            auth_manager = create_auth_manager()
            if auth_manager:
                st.session_state.spotify_auth_manager = auth_manager
                auth_url = get_auth_url(auth_manager)
                st.link_button(
                    "Connect to Spotify",
                    auth_url,
                    type="primary",
                    use_container_width=True,
                )

            st.markdown("""
            <div style="text-align: center; padding: 24px 0;">
                <span style="color: #535353;">─────── or ───────</span>
            </div>
            """, unsafe_allow_html=True)

        # === TOKEN-BASED APPROACH ===
        st.markdown("""
        <div style="
            background: #181818;
            border: 1px solid #282828;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 20px;
        ">
            <h4 style="margin-top: 0; color: #1DB954;">Get Your Spotify Data</h4>
            <p style="font-size: 0.9rem; color: #b3b3b3; margin-bottom: 0;">
                Follow these steps to export your listening history
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Fun explainer at the top
        with st.expander("Why can't I just one-click authorize?"):
            # Show images side by side using st.image (works reliably)
            spotify_images = [
                Path("assets/spotify1.jpeg"),
                Path("assets/spotify2.webp"),
                Path("assets/spotify3.webp"),
            ]
            existing_images = [str(p) for p in spotify_images if p.exists()]
            if existing_images:
                cols = st.columns(len(existing_images))
                for col, img in zip(cols, existing_images):
                    col.image(img, use_container_width=True)

            st.markdown("""
**A brief history of how we got here:**

**1986** — Geoffrey Hinton publishes "Learning representations by back-propagating errors" in Nature. Neural nets can learn.

**2012** — Ilya Sutskever has a vision:
> *"Then the scaling insight arrived. Scaling laws, GPT-3, and suddenly everyone realized we should scale. This is an example of how language affects thought. 'Scaling' is just one word, but it's such a powerful word."*

**2023** — The New York Times sues OpenAI for training on their articles. Sam Altman cannot be reached for comments, he's buying another yacht.

**Dec 2024** — Anna's Archive scrapes 86 million music files from Spotify and 256 million rows of metadata. Spotify's legal team panics! Unlike OpenAI, they cannot bankroll indefinite copyright lawsuits.

**Jan 2025** — Spotify freezes all new developer account creation. *"New integrations are on hold while we make updates to improve reliability."* No new Spotify apps allowed. Music fans in shambles.

**Now** — Claude Code helps you use bootleg Spotify apps without developer accounts.
            """)

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        # Step 1: Go to Spotify and get token
        st.markdown("### Step 1: Get a Token from Spotify")

        st.link_button(
            "Open Spotify API Page",
            "https://developer.spotify.com/documentation/web-api/reference/get-users-top-artists-and-tracks",
            type="primary",
            use_container_width=True,
        )

        # Show screenshot of where to find the token
        token_help_image = Path("assets/devtools_token.png")
        if token_help_image.exists():
            st.image(str(token_help_image), caption="Where to find the token in DevTools")

        with st.expander("Chrome / Edge", expanded=True):
            st.markdown("""
            1. **Right-click** anywhere on the Spotify page → **Inspect** (or press **F12**)
            2. Click the **Network** tab at the top of DevTools
            3. **Click "Try it"** on the Spotify page - this generates the token! (log in if prompted)
            4. In the Network list, click the **"artists"** request
            5. Click the **Headers** tab on the right panel
            6. Scroll down to **Request Headers** → find **Authorization**
            7. Copy **only the token part** (starts with `BQC...` or `eyJ...`) — **don't include the word "Bearer"**

            **Copy it before leaving the page!**
            """)

        with st.expander("Safari"):
            st.markdown("""
            1. Enable developer tools: Safari → Settings → Advanced → "Show Develop menu"
            2. **Develop menu** → **Show Web Inspector** (or **Cmd+Option+I**)
            3. Click the **Network** tab
            4. **Click "Try it"** on the Spotify page - this generates the token! (log in if prompted)
            5. Click the **artists** request in the list
            6. Find **Authorization: Bearer BQC...**
            7. Copy **only the token** (everything after "Bearer ") — don't include the word "Bearer"
            """)

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        # Step 2: Paste token
        st.markdown("### Step 2: Paste Your Token")

        token = st.text_input(
            "Spotify Access Token",
            type="password",
            placeholder="eyJ...",
            key="spotify_token",
            help="The token from the Authorization header",
        )

        if st.button("Fetch My Data", type="primary", use_container_width=True, disabled=not token):
            if token:
                _fetch_spotify_data_with_token(token)

        # Fallback: File upload
        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

        with st.expander("Alternative: Upload JSON file"):
            st.markdown("If you have a `spotify_data.json` file from a previous export:")

            uploaded_file = st.file_uploader(
                "Upload JSON file",
                type=["json"],
                key="spotify_upload",
            )

            if uploaded_file is not None:
                _process_uploaded_file(uploaded_file)


def render_artist_list(artists: list[Artist], songs: dict[str, list[Song]]) -> None:
    """Render the list of loaded artists."""
    st.subheader(f"Loaded Artists ({len(artists)})")

    for artist in artists:
        song_count = len(songs.get(artist.id, []))
        genres_str = ", ".join(artist.genres[:3]) if artist.genres else "No genres"
        st.write(f"**{artist.name}** - {song_count} songs - {genres_str}")


def _render_song_card(song: Song, artist: Artist, side: str) -> None:
    """Render a Spotify-style song card with embedded player."""
    # Use Spotify embed player - works without API key
    # Compact player with dark theme
    embed_url = f"https://open.spotify.com/embed/track/{song.id}?utm_source=generator&theme=0"

    st.markdown(f"""
    <div style="
        background: linear-gradient(145deg, #282828, #1a1a1a);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    ">
        <iframe
            src="{embed_url}"
            width="100%"
            height="152"
            frameBorder="0"
            allowfullscreen=""
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
            loading="lazy"
            style="border-radius: 12px;"
        ></iframe>
    </div>
    """, unsafe_allow_html=True)


def render_comparison_page(
    song_a: Song,
    song_b: Song,
    artist_a: Artist,
    artist_b: Artist,
) -> ComparisonResult | None:
    """Render a single comparison with Spotify-style cards.

    Returns:
        The user's choice, or None if no choice made yet
    """
    # Minimal header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 24px;">
        <span style="color: #b3b3b3; font-size: 14px;">Which do you prefer?</span>
    </div>
    """, unsafe_allow_html=True)

    # Two song cards side by side
    col1, spacer, col2 = st.columns([5, 1, 5])

    with col1:
        _render_song_card(song_a, artist_a, "left")

    with spacer:
        st.markdown("""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            height: 200px;
            color: #535353;
            font-size: 24px;
            font-weight: bold;
        ">VS</div>
        """, unsafe_allow_html=True)

    with col2:
        _render_song_card(song_b, artist_b, "right")

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # Choice buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(f"Prefer {artist_a.name}", type="primary", use_container_width=True):
            return ComparisonResult.PREFER_A

    with col2:
        if st.button("Can't Compare", use_container_width=True):
            return ComparisonResult.ABSTAIN

    with col3:
        if st.button(f"Prefer {artist_b.name}", type="primary", use_container_width=True):
            return ComparisonResult.PREFER_B

    # Help text
    st.caption("Choose 'Can't Compare' if these artists feel like they're doing fundamentally different things (e.g. high energy vs low energy) — not because you're unsure.")

    return None


def render_progress_sidebar(
    comparisons: list[Comparison],
    total_pairs: int,
    ranking_confidence: dict | None = None,
) -> None:
    """Render progress information in the sidebar."""
    st.sidebar.header("Progress")

    n_comparisons = len(comparisons)

    # Show milestone info prominently at top
    if ranking_confidence:
        status = ranking_confidence.get("status", "warming_up")
        to_next = ranking_confidence.get("comparisons_to_next", 0)

        if status == "warming_up":
            st.sidebar.markdown(f"### {to_next} to go")
            st.sidebar.caption("until initial ranking")
        elif status == "early":
            st.sidebar.markdown(f"### {to_next} to go")
            st.sidebar.caption("until ranking stabilizes")
        elif status == "refining":
            st.sidebar.markdown(f"### {to_next} to go")
            st.sidebar.caption("until solid ranking")
        else:
            st.sidebar.markdown("### Ranking solid!")
            st.sidebar.caption("Keep going to refine further")

        st.sidebar.divider()

    # Simple progress bar (don't show intimidating total)
    progress = n_comparisons / total_pairs if total_pairs > 0 else 0
    st.sidebar.progress(progress)
    st.sidebar.caption(f"{n_comparisons} comparisons made")

    # Count by result type
    prefer_a = sum(1 for c in comparisons if c.result == ComparisonResult.PREFER_A)
    prefer_b = sum(1 for c in comparisons if c.result == ComparisonResult.PREFER_B)
    abstain = sum(1 for c in comparisons if c.result == ComparisonResult.ABSTAIN)

    st.sidebar.write(f"- Preferences: {prefer_a + prefer_b}")
    st.sidebar.write(f"- Abstentions: {abstain}")


def render_confidence_banner(confidence: dict | None) -> None:
    """Render the ranking confidence banner with milestone markers."""
    if confidence is None:
        return

    status = confidence["status"]
    status_label = confidence["status_label"]
    percent = confidence["percent_complete"]
    to_next = confidence["comparisons_to_next"]

    # Banner message based on stage with specific milestones
    if status == "warming_up":
        msg = f"**{status_label}** — {to_next} to go until initial ranking"
        st.info(msg)
    elif status == "early":
        msg = f"**{status_label}** — {to_next} to go until ranking stabilizes"
        st.info(msg)
    elif status == "refining":
        msg = f"**{status_label}** — {to_next} to go until solid ranking"
        st.success(msg)
    else:  # solid
        msg = f"**{status_label}** — ranking is reliable ({percent:.0f}% of pairs compared)"
        st.success(msg)


def render_results_page(
    artists: list[Artist],
    posterior_summary: dict[str, dict[str, float]],
    k_dims: int = 1,
    ranking_confidence: dict | None = None,
    quadrant_labels: dict[str, str] | None = None,
) -> None:
    """Render the results visualization page."""
    from src.viz import (
        plot_genre_quadrants,
        plot_genre_space_1d,
        plot_genre_space_2d,
        plot_utility_ranking,
    )

    st.header("Results")

    # Confidence banner
    render_confidence_banner(ranking_confidence)

    # Genre space (Music Taste Map) - show first
    if k_dims == 1:
        st.subheader("Genre Space")
        fig_genre = plot_genre_space_1d(artists, posterior_summary)
    else:
        # Use quadrant visualization for 2D (with or without LLM labels)
        fig_genre = plot_genre_quadrants(artists, posterior_summary, quadrant_labels)
    st.plotly_chart(fig_genre, use_container_width=True)

    # Utility ranking - show below
    st.subheader("Artist Preference Ranking")
    fig_ranking = plot_utility_ranking(artists, posterior_summary)
    st.plotly_chart(fig_ranking, use_container_width=True)


def render_export_page(
    artists: list[Artist],
    songs: dict[str, list[Song]],
    comparisons: list[Comparison],
) -> None:
    """Render export options."""
    from src.data import SessionState

    st.header("Export Session")

    session = SessionState(
        artists=artists,
        songs=songs,
        comparisons=comparisons,
    )

    json_str = session.model_dump_json(indent=2)

    st.download_button(
        label="Download Session JSON",
        data=json_str,
        file_name="artist_ranking_session.json",
        mime="application/json",
    )

    st.subheader("Comparisons Made")
    for c in comparisons:
        artist_a = next((a for a in artists if a.id == c.artist_a_id), None)
        artist_b = next((a for a in artists if a.id == c.artist_b_id), None)
        if artist_a and artist_b:
            result_str = {
                ComparisonResult.PREFER_A: f"→ {artist_a.name}",
                ComparisonResult.PREFER_B: f"→ {artist_b.name}",
                ComparisonResult.ABSTAIN: "→ Can't Compare",
            }[c.result]
            st.write(f"{artist_a.name} vs {artist_b.name} {result_str}")
