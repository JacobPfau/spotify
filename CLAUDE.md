# Artist Preference Ranking System

## Project Overview

A Bayesian preference learning system that infers partial orderings over musical artists from pairwise song comparisons. Users compare two songs and indicate preference or abstain when artists feel incomparable (different genres/vibes). The model learns both a latent "genre space" positioning and preference utilities for each artist.

**Key insight**: Abstentions are informative—they signal genre distance, not uncertainty.

See `spec.md` for full mathematical model specification.

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key for LLM priors (optional but recommended)
export ANTHROPIC_API_KEY=your_key_here

# Run the app
streamlit run app.py
```

## Current Status

**Phase 1 (MVP): COMPLETE**
- [x] JSON parsing with filtering (loads from `data/` directory)
- [x] Streamlit UI (setup, compare, results, export pages)
- [x] Claude Sonnet integration for genre distance priors
- [x] MDS projection for prior means (1D and 2D)
- [x] NumPyro model with NUTS inference
- [x] Visualizations (utility ranking, genre space plots)
- [x] Random pair selection
- [x] K=1 and K=2 genre dimensions

**Phase 2 (Intelligence): COMPLETE**
- [x] Genre space visualization (1D and 2D)
- [x] Active selection (information gain) - uses posterior to pick most informative pairs

**Phase 3 (Polish): TODO**
- [ ] Comparability heatmap visualization
- [ ] Audio preview playback (when Spotify provides preview URLs)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  data/*.json    │────▶│  Streamlit UI    │────▶│  NumPyro Model  │
│  (file input)   │     │  (comparisons)   │     │  (inference)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  Claude Sonnet   │     │  Visualization  │
                        │  (genre priors)  │     │  (results)      │
                        └──────────────────┘     └─────────────────┘
```

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | Standard for ML |
| Inference | NumPyro | Fast JAX-based MCMC, good for active learning loops |
| UI | Streamlit | Rapid prototyping, sufficient for single-user |
| Visualization | Plotly | Interactive genre space / utility plots |
| LLM Prior | Claude Sonnet via API | Genre distance matrix for prior initialization |

## Input Data Format

Place Spotify API JSON files in the `data/` directory:
- `data/top_artists.json` - from "Get User's Top Artists" endpoint
- `data/top_tracks_1.json`, `data/top_tracks_2.json`, etc. - from "Get User's Top Tracks" (paginated, 50 per file)

### Data Preprocessing

1. Parse top N artists (configurable, default 25) → extract `{id, name, genres}`
2. Parse all tracks → filter to only tracks where primary artist is in the top N
3. Group songs by artist
4. Drop any artists with zero songs after filtering

## Core Modules

### `data.py` - Data Models & Parsing
- Pydantic models for Artist, Song, Comparison
- JSON parsing from Spotify format
- `load_from_data_dir()` - loads from data/ directory
- Filtering logic for track→artist matching

### `model.py` - NumPyro Model
- Implements hierarchical model from spec.md
- Genre positions `g_a` with anchor constraint (first artist at origin)
- Artist utilities `s_a` with anchor constraint
- Artist consistency `λ_a` (song quality variance)
- Comparability function with lengthscale `τ`
- NUTS inference via `run_inference()`

### `priors.py` - LLM Prior Generation
- `get_genre_priors_from_llm_sync()` - calls Claude Sonnet API
- Queries pairwise genre distances (1-5 scale) for all artist pairs
- Fits K-dimensional MDS to distance matrix
- Returns prior means for genre positions

### `selection.py` - Active Comparison Selection
- `compute_pair_score()` - expected information gain
- `select_next_pair()` - picks highest-scoring uncompared pair
- `select_random_pair()` - fallback before inference runs
- `select_songs_for_comparison()` - random song from each artist

### `ui.py` - Streamlit Interface
Pages/sections:
1. **Setup**: Load from data/, configure max artists, K dims, LLM priors
2. **Compare**: Present two songs, three buttons (A/B/Can't Compare)
3. **Results**: Genre space visualization, utility rankings with uncertainty
4. **Export**: Download comparisons as JSON

### `viz.py` - Visualization
- `plot_utility_ranking()` - horizontal bar chart with error bars
- `plot_genre_space_1d()` - scatter with genre (x) vs utility (y)
- `plot_genre_space_2d()` - 2D genre map with utility as color
- `plot_comparability_heatmap()` - pairwise comparability matrix

## Session State

```python
st.session_state.artists: list[Artist]
st.session_state.songs: dict[str, list[Song]]  # artist_id -> songs
st.session_state.comparisons: list[Comparison]
st.session_state.posterior: dict | None  # latest inference summary
st.session_state.prior_genre_means: np.ndarray | None  # from LLM
st.session_state.k_dims: int  # 1 or 2
st.session_state.samples: dict | None  # raw MCMC samples
```

## Inference Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| K (genre dims) | 1 or 2 | User selectable at setup |
| MCMC warmup | 500 | Reduced for faster iteration |
| MCMC samples | 500 | Sufficient for ~25 artists |
| Chains | 1 | Single chain for speed (TODO: add convergence checks) |
| Re-run trigger | Manual | "Update Model" button in sidebar |

## File Structure

```
spotify-tournament/
├── CLAUDE.md           # This file
├── spec.md             # Mathematical specification
├── requirements.txt
├── app.py              # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── data.py         # Models, parsing, load_from_data_dir
│   ├── model.py        # NumPyro model, run_inference
│   ├── priors.py       # LLM prior generation
│   ├── selection.py    # Active learning (+ random fallback)
│   ├── viz.py          # Plotly visualizations
│   └── ui.py           # Streamlit components
├── data/               # User's Spotify JSON files
│   ├── top_artists.json
│   ├── top_tracks_1.json
│   ├── top_tracks_2.json
│   └── top_tracks_3.json
└── tests/
    ├── __init__.py
    ├── test_data.py
    ├── test_model.py
    └── fixtures/
        └── sample_spotify.json
```

## Key Implementation Notes

### Identifiability
First artist is anchor: `g[0] = 0`, `s[0] = 0`. All positions/utilities are relative.

### Song Selection
When comparing artists A vs B, a random song is selected from each artist's catalog. The specific songs compared are recorded for potential future analysis.

### Abstention Semantics
UI explains that "Can't Compare" means "these artists are doing fundamentally different things" NOT "I'm unsure which I prefer."

### Incremental Inference
User-triggered via "Update Model" button. Does not auto-run to avoid blocking UI.

## Remaining Polish Tasks

1. **Convergence Diagnostics**: Display r-hat and ESS warnings when model may not have converged
2. **Session Import**: Add file upload to restore previous session JSON
3. **Comparability Heatmap**: Add to results page using `plot_comparability_heatmap()`
4. **Loading States**: Better spinners/progress for inference
5. **Edge Cases**: Handle artists with very few songs, empty comparisons gracefully

## Future Extensions

- Use as ranking prior the artist play rate
- Improve visual experience: To compete with spotify wrapped-like experience
- Direct Spotify API integration (OAuth flow)
- Audio feature integration as additional priors
