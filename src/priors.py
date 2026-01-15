"""LLM-based prior generation for genre distances."""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.manifold import MDS

from src.data import Artist


def _get_anthropic_api_key() -> str | None:
    """Get Anthropic API key from Streamlit secrets or environment."""
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

    # Fall back to environment variable
    return os.environ.get("ANTHROPIC_API_KEY")


def build_distance_prompt(artists: list[Artist]) -> str:
    """Build the prompt for querying artist genre distances."""
    artist_list = "\n".join(f"- {a.name}" for a in artists)

    prompt = f"""Given these musical artists:
{artist_list}

Rate the musical/genre distance between each pair on a 1-5 scale:
1 = very similar (same genre, similar style)
2 = somewhat similar (related genres, compatible styles)
3 = moderately different (different genres but some overlap)
4 = quite different (distinct genres, little overlap)
5 = completely different (opposite ends of musical spectrum)

Return your response as JSON with this exact format:
{{"pairs": [
  {{"a": "Artist1", "b": "Artist2", "distance": 3}},
  ...
]}}

Include ALL unique pairs (for N artists, that's N*(N-1)/2 pairs).
Only return the JSON, no other text."""

    return prompt


def parse_distance_response(
    response: str,
    artists: list[Artist],
) -> np.ndarray:
    """Parse the LLM response into a distance matrix.

    Args:
        response: JSON string from LLM
        artists: List of artists

    Returns:
        n_artists x n_artists distance matrix
    """
    # Try to extract JSON from response
    try:
        # Handle case where response might have markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        data = json.loads(response.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    # Build name -> index mapping
    name_to_idx = {a.name.lower(): i for i, a in enumerate(artists)}

    n = len(artists)
    distances = np.zeros((n, n))

    pairs = data.get("pairs", [])
    for pair in pairs:
        name_a = pair["a"].lower()
        name_b = pair["b"].lower()
        dist = pair["distance"]

        # Try to find matching artists (handle slight name variations)
        idx_a = name_to_idx.get(name_a)
        idx_b = name_to_idx.get(name_b)

        if idx_a is None:
            # Try partial match
            for name, idx in name_to_idx.items():
                if name_a in name or name in name_a:
                    idx_a = idx
                    break

        if idx_b is None:
            for name, idx in name_to_idx.items():
                if name_b in name or name in name_b:
                    idx_b = idx
                    break

        if idx_a is not None and idx_b is not None:
            distances[idx_a, idx_b] = dist
            distances[idx_b, idx_a] = dist

    return distances


def fit_mds_positions(
    distances: np.ndarray,
    k_dims: int = 1,
    random_state: int = 42,
) -> np.ndarray:
    """Fit MDS to distance matrix to get genre positions.

    Args:
        distances: n_artists x n_artists distance matrix
        k_dims: Number of dimensions for MDS embedding
        random_state: Random seed for reproducibility

    Returns:
        n_artists x k_dims array of genre positions
    """
    # MDS expects dissimilarities - our 1-5 scale works directly
    mds = MDS(
        n_components=k_dims,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress="auto",
    )

    positions = mds.fit_transform(distances)

    # Center the positions (MDS output is already centered, but ensure it)
    positions = positions - positions.mean(axis=0)

    # Scale to unit variance per dimension (to match prior scale assumption)
    std = positions.std(axis=0)
    std = np.where(std > 0, std, 1.0)  # Avoid division by zero
    positions = positions / std

    return positions


async def get_genre_priors_from_llm(
    artists: list[Artist],
    k_dims: int = 1,
) -> np.ndarray:
    """Query LLM for genre distances and compute MDS positions.

    Args:
        artists: List of artists
        k_dims: Number of genre dimensions

    Returns:
        n_artists x k_dims array of prior genre position means
    """
    import anthropic

    client = anthropic.Anthropic()

    prompt = build_distance_prompt(artists)

    message = await client.messages.create(
        model="claude-sonnet-4-5-latest",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text

    # Parse distances
    distances = parse_distance_response(response_text, artists)

    # Fit MDS
    positions = fit_mds_positions(distances, k_dims=k_dims)

    return positions


def get_genre_priors_from_llm_sync(
    artists: list[Artist],
    k_dims: int = 1,
) -> np.ndarray:
    """Synchronous version of get_genre_priors_from_llm."""
    import anthropic

    client = anthropic.Anthropic()

    prompt = build_distance_prompt(artists)

    message = client.messages.create(
        model="claude-sonnet-4-5-latest",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text

    # Parse distances
    distances = parse_distance_response(response_text, artists)

    # Fit MDS
    positions = fit_mds_positions(distances, k_dims=k_dims)

    return positions


def get_default_priors(n_artists: int, k_dims: int = 1) -> np.ndarray:
    """Return default (zero) priors when LLM is unavailable."""
    return np.zeros((n_artists, k_dims))


def get_quadrant_labels_from_llm(
    artists_by_quadrant: dict[str, list[str]],
) -> dict[str, str]:
    """Generate 2-3 word labels for each quadrant based on artists in it.

    Args:
        artists_by_quadrant: Dict mapping quadrant names ("top_left", "top_right",
            "bottom_left", "bottom_right") to lists of artist names

    Returns:
        Dict mapping quadrant names to short descriptive labels (2-3 words)
    """
    import anthropic

    # Build the prompt
    quadrant_descriptions = []
    for quadrant, artists in artists_by_quadrant.items():
        if artists:
            artist_str = ", ".join(artists[:5])  # Limit to 5 artists per quadrant
            quadrant_descriptions.append(f"- {quadrant}: {artist_str}")

    if not quadrant_descriptions:
        return {
            "top_left": "Quadrant 1",
            "top_right": "Quadrant 2",
            "bottom_left": "Quadrant 3",
            "bottom_right": "Quadrant 4",
        }

    quadrant_text = "\n".join(quadrant_descriptions)

    prompt = f"""Given these musical artists grouped by quadrant in a 2D genre space:

{quadrant_text}

Generate a short, evocative label (2-3 words max) for each quadrant that captures the musical vibe/aesthetic of the artists in it.

Labels should be descriptive and fun - think music magazine vibes, not academic genres.
Examples of good labels: "Indie Dreams", "Electronic Pulse", "Raw Energy", "Smooth Grooves", "Dark Atmospheres"

Return your response as JSON with this exact format:
{{"top_left": "Label 1", "top_right": "Label 2", "bottom_left": "Label 3", "bottom_right": "Label 4"}}

Only return the JSON, no other text."""

    client = anthropic.Anthropic()

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-latest",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text

        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        labels = json.loads(response_text.strip())

        # Ensure all quadrants have labels
        default_labels = {
            "top_left": "Quadrant 1",
            "top_right": "Quadrant 2",
            "bottom_left": "Quadrant 3",
            "bottom_right": "Quadrant 4",
        }
        for q in default_labels:
            if q not in labels or not labels[q]:
                labels[q] = default_labels[q]

        return labels

    except Exception:
        return {
            "top_left": "Quadrant 1",
            "top_right": "Quadrant 2",
            "bottom_left": "Quadrant 3",
            "bottom_right": "Quadrant 4",
        }
