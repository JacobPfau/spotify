"""Streamlit entry point for the Artist Preference Ranking app."""

import streamlit as st

from src.data import Artist, Comparison, ComparisonResult, Song, get_artist_pairs
from src.selection import select_next_pair, select_random_pair, select_songs_for_comparison
from src.ui import (
    render_artist_list,
    render_comparison_page,
    render_export_page,
    render_progress_sidebar,
    render_results_page,
    render_setup_page,
)

st.set_page_config(
    page_title="Artist Preference Ranking",
    page_icon="🎵",
    layout="wide",
)

# Apply Spotify-inspired dark theme CSS
st.markdown("""
<style>
    /* === Global Dark Theme === */
    .stApp {
        background: linear-gradient(180deg, #1a1a1a 0%, #121212 100%);
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* === Typography === */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    h1 {
        font-size: 2.5rem !important;
        letter-spacing: -0.02em !important;
    }

    p, .stMarkdown, label {
        color: #b3b3b3 !important;
    }

    /* === Sidebar Styling === */
    [data-testid="stSidebar"] {
        background: #000000 !important;
        border-right: 1px solid #282828;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #b3b3b3 !important;
    }

    /* Sidebar navigation radio buttons */
    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        background: transparent !important;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 2px 0;
        transition: background 0.2s ease;
    }

    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background: #282828 !important;
    }

    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background: #282828 !important;
    }

    /* === Buttons === */
    .stButton > button {
        background: #282828 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 500px !important;
        padding: 12px 32px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #3e3e3e !important;
        transform: scale(1.02);
    }

    /* Primary buttons - Spotify green with black text */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"],
    button[kind="primary"],
    [data-testid="baseButton-primary"] {
        background: #1DB954 !important;
        background-color: #1DB954 !important;
        color: #000000 !important;
    }

    .stButton > button[kind="primary"] p,
    .stButton > button[data-testid="baseButton-primary"] p,
    button[kind="primary"] p {
        color: #000000 !important;
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover,
    button[kind="primary"]:hover {
        background: #1ed760 !important;
        background-color: #1ed760 !important;
        color: #000000 !important;
    }

    /* Link buttons with primary styling */
    .stLinkButton > a[data-testid="baseLinkButton-primary"],
    a[kind="primary"] {
        background: #1DB954 !important;
        color: #000000 !important;
    }

    .stLinkButton > a[data-testid="baseLinkButton-primary"] p,
    .stLinkButton > a[data-testid="baseLinkButton-primary"] span,
    a[kind="primary"] p,
    a[kind="primary"] span {
        color: #000000 !important;
    }

    .stLinkButton > a[data-testid="baseLinkButton-primary"]:hover,
    a[kind="primary"]:hover {
        background: #1ed760 !important;
        color: #000000 !important;
    }

    /* === Progress Bar === */
    .stProgress > div > div {
        background: #535353 !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%) !important;
    }

    /* === Cards & Containers === */
    .stExpander {
        background: #181818 !important;
        border: 1px solid #282828 !important;
        border-radius: 8px !important;
    }

    .stExpander summary {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* === Alerts & Messages === */
    .stAlert {
        border-radius: 8px !important;
        border: none !important;
    }

    /* Info alert */
    [data-testid="stAlert"][data-type="info"] {
        background: rgba(29, 185, 84, 0.1) !important;
        border-left: 4px solid #1DB954 !important;
    }

    /* Success alert */
    [data-testid="stAlert"][data-type="success"] {
        background: rgba(29, 185, 84, 0.15) !important;
        border-left: 4px solid #1DB954 !important;
    }

    /* Warning alert */
    [data-testid="stAlert"][data-type="warning"] {
        background: rgba(255, 193, 7, 0.1) !important;
        border-left: 4px solid #ffc107 !important;
    }

    /* Error alert */
    [data-testid="stAlert"][data-type="error"] {
        background: rgba(255, 82, 82, 0.1) !important;
        border-left: 4px solid #ff5252 !important;
    }

    /* === Input Fields === */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #282828 !important;
        color: #ffffff !important;
        border: 1px solid #535353 !important;
        border-radius: 4px !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #1DB954 !important;
        box-shadow: 0 0 0 1px #1DB954 !important;
    }

    /* === Sliders === */
    .stSlider > div > div > div > div {
        background: #1DB954 !important;
    }

    /* === Select boxes & Radio === */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #282828 !important;
        border-color: #535353 !important;
    }

    /* === Dividers === */
    hr {
        border-color: #282828 !important;
    }

    /* === Links === */
    a {
        color: #1DB954 !important;
    }

    a:hover {
        color: #1ed760 !important;
    }

    /* === Captions === */
    .stCaption, caption, .caption {
        color: #727272 !important;
    }

    /* === Tab styling === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 4px;
        color: #b3b3b3 !important;
        padding: 8px 16px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
    }

    .stTabs [aria-selected="true"] {
        background: #282828 !important;
        color: #ffffff !important;
    }

    /* === Metrics === */
    [data-testid="stMetric"] {
        background: #181818;
        padding: 16px;
        border-radius: 8px;
    }

    [data-testid="stMetric"] label {
        color: #b3b3b3 !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* === Spinner === */
    .stSpinner > div {
        border-color: #1DB954 transparent transparent transparent !important;
    }

    /* === Checkbox === */
    .stCheckbox > label > span {
        color: #b3b3b3 !important;
    }

    /* === Plotly charts dark background === */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }

    .js-plotly-plot .plotly .modebar-btn path {
        fill: #b3b3b3 !important;
    }

    /* === Custom scrollbar === */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }

    ::-webkit-scrollbar-track {
        background: #121212;
    }

    ::-webkit-scrollbar-thumb {
        background: #535353;
        border-radius: 6px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #727272;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize session state variables."""
    if "artists" not in st.session_state:
        st.session_state.artists = []
    if "songs" not in st.session_state:
        st.session_state.songs = {}
    if "comparisons" not in st.session_state:
        st.session_state.comparisons = []
    if "posterior" not in st.session_state:
        st.session_state.posterior = None
    if "prior_genre_means" not in st.session_state:
        st.session_state.prior_genre_means = None
    if "k_dims" not in st.session_state:
        st.session_state.k_dims = 1
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False
    if "current_pair" not in st.session_state:
        st.session_state.current_pair = None
    if "current_songs" not in st.session_state:
        st.session_state.current_songs = None
    if "inference_running" not in st.session_state:
        st.session_state.inference_running = False
    if "last_inference_count" not in st.session_state:
        st.session_state.last_inference_count = 0
    if "ranking_confidence" not in st.session_state:
        st.session_state.ranking_confidence = None
    if "quadrant_labels" not in st.session_state:
        st.session_state.quadrant_labels = None
    if "quadrant_labels_generated" not in st.session_state:
        st.session_state.quadrant_labels_generated = False
    if "pending_inference" not in st.session_state:
        st.session_state.pending_inference = False
    # Spotify OAuth state
    if "spotify_auth_manager" not in st.session_state:
        st.session_state.spotify_auth_manager = None
    if "spotify_connected" not in st.session_state:
        st.session_state.spotify_connected = False


def handle_spotify_callback() -> bool:
    """Handle Spotify OAuth callback if code is in query params.

    Returns:
        True if callback was handled and data was fetched
    """
    # Check for OAuth callback code
    query_params = st.query_params
    code = query_params.get("code")

    if not code:
        return False

    # We have a code - try to exchange it for a token
    auth_manager = st.session_state.spotify_auth_manager

    if auth_manager is None:
        # Create a new auth manager if needed
        from src.spotify import create_auth_manager
        auth_manager = create_auth_manager()
        if auth_manager is None:
            st.error("Failed to create Spotify auth manager")
            st.query_params.clear()
            return False

    # Exchange code for token
    from src.spotify import exchange_code_for_token, get_spotify_client, fetch_spotify_data

    with st.spinner("Connecting to Spotify..."):
        token_info = exchange_code_for_token(auth_manager, code)

        if token_info is None:
            st.error("Failed to authenticate with Spotify. Please try again.")
            st.query_params.clear()
            return False

        # Get Spotify client
        sp = get_spotify_client(auth_manager)
        if sp is None:
            st.error("Failed to create Spotify client")
            st.query_params.clear()
            return False

        # Fetch data
        try:
            time_range = st.session_state.get("spotify_time_range", "medium_term")
            artists_json, tracks_json = fetch_spotify_data(
                sp,
                max_artists=50,  # Fetch up to 50, user can filter later
                max_tracks=250,
                time_range=time_range,
            )

            # Store raw data for processing in setup page
            st.session_state.spotify_artists_json = artists_json
            st.session_state.spotify_tracks_json = tracks_json
            st.session_state.spotify_connected = True
            st.session_state.spotify_auth_manager = auth_manager

            # Clear query params to avoid re-processing
            st.query_params.clear()

            return True

        except Exception as e:
            st.error(f"Failed to fetch Spotify data: {e}")
            st.query_params.clear()
            return False

    return False


def get_next_comparison() -> tuple[Song, Song, Artist, Artist] | None:
    """Get the next pair of songs to compare."""
    artists: list[Artist] = st.session_state.artists
    songs: dict[str, list[Song]] = st.session_state.songs
    comparisons: list[Comparison] = st.session_state.comparisons

    # Use active selection if we have posterior samples, otherwise random
    samples = st.session_state.get("samples")
    if samples is not None:
        artist_id_to_idx = {a.id: i for i, a in enumerate(artists)}
        pair = select_next_pair(
            artists=artists,
            comparisons=comparisons,
            samples=samples,
            artist_id_to_idx=artist_id_to_idx,
            top_k=3,  # Sample from top 3 for some randomization
        )
    else:
        # Random selection before first inference
        pair = select_random_pair(artists, comparisons)

    if pair is None:
        return None

    artist_a_id, artist_b_id = pair
    artist_a = next(a for a in artists if a.id == artist_a_id)
    artist_b = next(a for a in artists if a.id == artist_b_id)

    song_a, song_b = select_songs_for_comparison(artist_a_id, artist_b_id, songs)

    return song_a, song_b, artist_a, artist_b


def _run_inference_sync(
    n_artists: int,
    comparisons: list[Comparison],
    artist_id_to_idx: dict[str, int],
    artist_ids: list[str],
    prior_means,
    k_dims: int,
) -> tuple[dict, dict] | None:
    """Run inference synchronously. Returns (samples, posterior_summary) or None on error."""
    import traceback

    try:
        from src.model import extract_posterior_summary, run_inference
        import jax.numpy as jnp

        if prior_means is not None:
            prior_means = jnp.array(prior_means)

        samples = run_inference(
            n_artists=n_artists,
            comparisons=comparisons,
            artist_id_to_idx=artist_id_to_idx,
            prior_genre_means=prior_means,
            k_dims=k_dims,
            num_warmup=500,
            num_samples=500,
            num_chains=1,
        )

        posterior_summary = extract_posterior_summary(samples, artist_ids)
        return samples, posterior_summary

    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None


def run_inference_if_needed() -> None:
    """Trigger inference if we have enough comparisons and haven't run recently."""
    comparisons: list[Comparison] = st.session_state.comparisons
    artists: list[Artist] = st.session_state.artists
    n_comparisons = len(comparisons)

    # Only run inference after minimum comparisons
    min_comparisons = 10
    if n_comparisons < min_comparisons:
        return

    # Check if already running
    if st.session_state.inference_running:
        return

    # Run every 10 comparisons (after minimum)
    last_count = st.session_state.last_inference_count
    if n_comparisons - last_count < 10 and last_count > 0:
        return

    # Mark as running and update count
    st.session_state.inference_running = True
    st.session_state.last_inference_count = n_comparisons

    artist_id_to_idx = {a.id: i for i, a in enumerate(artists)}
    artist_ids = [a.id for a in artists]
    prior_means = st.session_state.prior_genre_means
    k_dims = st.session_state.k_dims

    # Run inference synchronously (Streamlit reruns make true background hard)
    # Use spinner for user feedback
    try:
        with st.spinner("Updating model..."):
            result = _run_inference_sync(
                n_artists=len(artists),
                comparisons=list(comparisons),  # Copy to avoid mutation issues
                artist_id_to_idx=artist_id_to_idx,
                artist_ids=artist_ids,
                prior_means=prior_means,
                k_dims=k_dims,
            )

            if result is not None:
                samples, posterior_summary = result
                st.session_state.samples = samples
                st.session_state.posterior = posterior_summary

                # Compute ranking confidence
                from src.model import compute_ranking_confidence
                total_pairs = len(get_artist_pairs(artists))
                confidence = compute_ranking_confidence(
                    n_comparisons=n_comparisons,
                    total_pairs=total_pairs,
                )
                st.session_state.ranking_confidence = confidence

                # Generate quadrant labels once at "usable" threshold for 2D mode
                if (
                    k_dims == 2
                    and not st.session_state.quadrant_labels_generated
                    and confidence["status"] in ("usable", "good", "final")
                ):
                    try:
                        from src.priors import get_quadrant_labels_from_llm
                        from src.viz import get_artists_by_quadrant

                        artists_by_quadrant = get_artists_by_quadrant(
                            artists, posterior_summary
                        )
                        quadrant_labels = get_quadrant_labels_from_llm(artists_by_quadrant)
                        st.session_state.quadrant_labels = quadrant_labels
                        st.session_state.quadrant_labels_generated = True
                    except Exception:
                        pass  # Use default labels if generation fails
    finally:
        st.session_state.inference_running = False


def main() -> None:
    """Main app entry point."""
    init_session_state()

    # Handle Spotify OAuth callback (must happen before rendering)
    handle_spotify_callback()

    st.markdown("""
    <h1 style="text-align: center; font-size: 2.5rem;">Artist Preference Ranking</h1>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    if st.session_state.setup_complete:
        page = st.sidebar.radio(
            "Navigation",
            ["Compare", "Results", "Export"],
            key="nav",
        )

        # Show progress - always recompute ranking confidence
        artists = st.session_state.artists
        comparisons = st.session_state.comparisons
        total_pairs = len(get_artist_pairs(artists))

        # Update ranking confidence on every render
        from src.model import compute_ranking_confidence
        st.session_state.ranking_confidence = compute_ranking_confidence(
            n_comparisons=len(comparisons),
            total_pairs=total_pairs,
        )

        render_progress_sidebar(
            comparisons,
            total_pairs,
            ranking_confidence=st.session_state.ranking_confidence,
        )

        # Option to reset
        if st.sidebar.button("Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        # Manual inference trigger (as backup, model now auto-updates)
        if st.sidebar.button("Force Update Model"):
            st.session_state.last_inference_count = 0  # Force update
            run_inference_if_needed()
            st.rerun()

    else:
        page = "Setup"

    # Render the appropriate page
    if page == "Setup":
        if render_setup_page():
            # Setup complete, show artist list
            render_artist_list(st.session_state.artists, st.session_state.songs)

    elif page == "Compare":
        artists = st.session_state.artists
        songs = st.session_state.songs
        comparisons = st.session_state.comparisons

        # Get or refresh current comparison
        if st.session_state.current_pair is None:
            result = get_next_comparison()
            if result is None:
                st.success("All pairs have been compared!")
                st.balloons()
            else:
                song_a, song_b, artist_a, artist_b = result
                st.session_state.current_pair = (artist_a, artist_b)
                st.session_state.current_songs = (song_a, song_b)

        if st.session_state.current_pair is not None:
            artist_a, artist_b = st.session_state.current_pair
            song_a, song_b = st.session_state.current_songs

            choice = render_comparison_page(song_a, song_b, artist_a, artist_b)

            if choice is not None:
                # Record comparison
                comparison = Comparison(
                    song_a_id=song_a.id,
                    song_b_id=song_b.id,
                    artist_a_id=artist_a.id,
                    artist_b_id=artist_b.id,
                    result=choice,
                )
                st.session_state.comparisons.append(comparison)

                # Clear current pair to get a new one
                st.session_state.current_pair = None
                st.session_state.current_songs = None

                # Flag that we should check for inference on next render
                # (don't block here - show next comparison first)
                st.session_state.pending_inference = True

                st.rerun()

        # Run deferred inference AFTER page is rendered
        # This still blocks but at least the comparison UI is already visible
        if st.session_state.pending_inference:
            st.session_state.pending_inference = False
            run_inference_if_needed()

    elif page == "Results":
        # Try to run inference if we have enough comparisons but no posterior yet
        comparisons = st.session_state.comparisons
        if st.session_state.posterior is None and len(comparisons) >= 10:
            # Force inference to run by resetting flags
            st.session_state.last_inference_count = 0
            st.session_state.inference_running = False  # Reset in case it's stuck
            run_inference_if_needed()

        if st.session_state.posterior is None:
            n_comparisons = len(st.session_state.comparisons)
            if n_comparisons < 10:
                st.info(
                    f"Make at least 10 comparisons to see results. You have {n_comparisons} so far."
                )
            else:
                st.warning(
                    "Model is still computing. Click 'Force Update Model' in the sidebar or make another comparison."
                )
        else:
            render_results_page(
                st.session_state.artists,
                st.session_state.posterior,
                k_dims=st.session_state.k_dims,
                ranking_confidence=st.session_state.ranking_confidence,
                quadrant_labels=st.session_state.quadrant_labels,
            )

    elif page == "Export":
        render_export_page(
            st.session_state.artists,
            st.session_state.songs,
            st.session_state.comparisons,
        )


if __name__ == "__main__":
    main()
