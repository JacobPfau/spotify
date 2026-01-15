"""Visualization functions using Plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

if TYPE_CHECKING:
    from jax import Array

from src.data import Artist


def _repel_overlapping_images(
    positions: list[tuple[float, float]],
    sizes: list[float],
    iterations: int = 50,
    repel_strength: float = 0.3,
) -> list[tuple[float, float]]:
    """Apply force-directed repulsion to reduce image overlap.

    Args:
        positions: List of (x, y) positions
        sizes: List of image sizes (used as radii)
        iterations: Number of repulsion iterations
        repel_strength: How strongly overlapping images repel

    Returns:
        Adjusted positions with reduced overlap
    """
    if len(positions) < 2:
        return positions

    positions = [list(p) for p in positions]  # Make mutable
    n = len(positions)

    for _ in range(iterations):
        # Calculate repulsion forces
        forces = [[0.0, 0.0] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                dist = np.sqrt(dx**2 + dy**2) + 1e-6

                # Minimum distance before overlap (sum of radii)
                min_dist = (sizes[i] + sizes[j]) / 2

                if dist < min_dist:
                    # Overlap - apply repulsion
                    overlap = min_dist - dist
                    force = overlap * repel_strength / dist

                    forces[i][0] -= dx * force
                    forces[i][1] -= dy * force
                    forces[j][0] += dx * force
                    forces[j][1] += dy * force

        # Apply forces
        for i in range(n):
            positions[i][0] += forces[i][0]
            positions[i][1] += forces[i][1]

    return [(p[0], p[1]) for p in positions]


def plot_utility_ranking(
    artists: list[Artist],
    posterior_summary: dict[str, dict[str, float]],
) -> go.Figure:
    """Create a horizontal bar chart of artist utilities with uncertainty.

    Args:
        artists: List of artists
        posterior_summary: Dictionary with posterior statistics per artist

    Returns:
        Plotly figure
    """
    # Sort artists by utility
    sorted_artists = sorted(
        artists,
        key=lambda a: posterior_summary[a.id]["utility_mean"],
        reverse=True,
    )

    names = [a.name for a in sorted_artists]
    utilities = [posterior_summary[a.id]["utility_mean"] for a in sorted_artists]
    errors = [posterior_summary[a.id]["utility_std"] for a in sorted_artists]

    # Color gradient: Spotify green for top, fading to gray
    n = len(sorted_artists)
    colors = []
    for i in range(n):
        if i == 0:
            colors.append("#1DB954")  # Spotify green for #1
        elif i < 3:
            colors.append("#1ed760")  # Lighter green for top 3
        elif i < n // 2:
            colors.append("#b3b3b3")  # Light gray for upper half
        else:
            colors.append("#535353")  # Dark gray for lower half

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=names,
            x=utilities,
            orientation="h",
            error_x=dict(
                type="data",
                array=errors,
                visible=True,
                color="#727272",
            ),
            marker=dict(
                color=colors,
                line=dict(color="#282828", width=1),
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Utility: %{x:.2f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Your Artist Ranking",
            font=dict(size=20, color="#ffffff"),
        ),
        xaxis=dict(
            title="Preference Score",
            title_font=dict(color="#b3b3b3"),
            tickfont=dict(color="#b3b3b3"),
            gridcolor="#282828",
            zerolinecolor="#535353",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color="#ffffff"),
            autorange="reversed",
        ),
        height=max(400, len(artists) * 28),
        showlegend=False,
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        hoverlabel=dict(
            bgcolor="#282828",
            font_size=14,
            font_color="#ffffff",
        ),
    )

    return fig


def plot_genre_space_1d(
    artists: list[Artist],
    posterior_summary: dict[str, dict[str, float]],
) -> go.Figure:
    """Create scatter plot with genre position (x) vs utility (y).

    Args:
        artists: List of artists
        posterior_summary: Dictionary with posterior statistics per artist

    Returns:
        Plotly figure
    """
    names = [a.name for a in artists]
    genre_positions = [posterior_summary[a.id]["genre_mean"][0] for a in artists]
    genre_errors = [posterior_summary[a.id]["genre_std"][0] for a in artists]
    utilities = [posterior_summary[a.id]["utility_mean"] for a in artists]
    utility_errors = [posterior_summary[a.id]["utility_std"] for a in artists]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=genre_positions,
            y=utilities,
            mode="markers+text",
            text=names,
            textposition="top center",
            error_x=dict(type="data", array=genre_errors, visible=True),
            error_y=dict(type="data", array=utility_errors, visible=True),
            marker=dict(size=12, color="steelblue"),
        )
    )

    fig.update_layout(
        title="Genre Space (1D) vs Utility",
        xaxis_title="Genre Position",
        yaxis_title="Utility",
        height=600,
        showlegend=False,
    )

    return fig


def plot_genre_space_2d(
    artists: list[Artist],
    posterior_summary: dict[str, dict[str, float]],
) -> go.Figure:
    """Create 2D scatter plot of genre space with utility as color.

    Args:
        artists: List of artists
        posterior_summary: Dictionary with posterior statistics per artist

    Returns:
        Plotly figure
    """
    names = [a.name for a in artists]
    genre_x = [posterior_summary[a.id]["genre_mean"][0] for a in artists]
    genre_y = [posterior_summary[a.id]["genre_mean"][1] for a in artists]
    utilities = [posterior_summary[a.id]["utility_mean"] for a in artists]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=genre_x,
            y=genre_y,
            mode="markers+text",
            text=names,
            textposition="top center",
            marker=dict(
                size=15,
                color=utilities,
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Utility"),
            ),
        )
    )

    fig.update_layout(
        title="Genre Space (2D)",
        xaxis_title="Genre Dimension 1",
        yaxis_title="Genre Dimension 2",
        height=600,
        showlegend=False,
    )

    return fig


def get_artists_by_quadrant(
    artists: list[Artist],
    posterior_summary: dict[str, dict[str, float]],
) -> dict[str, list[str]]:
    """Group artists by their quadrant in 2D genre space.

    Args:
        artists: List of artists
        posterior_summary: Dictionary with posterior statistics per artist

    Returns:
        Dict mapping quadrant names to lists of artist names
    """
    quadrants: dict[str, list[str]] = {
        "top_left": [],
        "top_right": [],
        "bottom_left": [],
        "bottom_right": [],
    }

    for artist in artists:
        genre_mean = posterior_summary[artist.id]["genre_mean"]
        x, y = genre_mean[0], genre_mean[1]

        if x < 0 and y >= 0:
            quadrants["top_left"].append(artist.name)
        elif x >= 0 and y >= 0:
            quadrants["top_right"].append(artist.name)
        elif x < 0 and y < 0:
            quadrants["bottom_left"].append(artist.name)
        else:
            quadrants["bottom_right"].append(artist.name)

    return quadrants


def plot_genre_quadrants(
    artists: list[Artist],
    posterior_summary: dict[str, dict[str, float]],
    quadrant_labels: dict[str, str] | None = None,
) -> go.Figure:
    """Create 2D genre space with quadrant labels, artist images, and ranking highlights.

    Features:
    - Dark theme matching Spotify aesthetic
    - Quadrant dividers at origin (0,0)
    - Artist images displayed (when available)
    - Marker size scaled by utility rank
    - Spotify green highlight on top-ranked artist
    - Quadrant labels in corners

    Args:
        artists: List of artists
        posterior_summary: Dictionary with posterior statistics per artist
        quadrant_labels: Optional dict with labels for each quadrant

    Returns:
        Plotly figure
    """
    if quadrant_labels is None:
        quadrant_labels = {
            "top_left": "Quadrant 1",
            "top_right": "Quadrant 2",
            "bottom_left": "Quadrant 3",
            "bottom_right": "Quadrant 4",
        }

    # Sort artists by utility to determine rankings
    sorted_artists = sorted(
        artists,
        key=lambda a: posterior_summary[a.id]["utility_mean"],
        reverse=True,
    )
    rank_map = {a.id: i for i, a in enumerate(sorted_artists)}

    # Get raw genre positions and scale them up for better spread
    raw_genre_x = [posterior_summary[a.id]["genre_mean"][0] for a in artists]
    raw_genre_y = [posterior_summary[a.id]["genre_mean"][1] for a in artists]

    # Scale factor to spread points apart - multiply coordinates by this
    SPREAD_MULTIPLIER = 6.0

    genre_x = [x * SPREAD_MULTIPLIER for x in raw_genre_x]
    genre_y = [y * SPREAD_MULTIPLIER for y in raw_genre_y]

    x_min, x_max = min(genre_x) - 0.5, max(genre_x) + 0.5
    y_min, y_max = min(genre_y) - 0.5, max(genre_y) + 0.5

    fig = go.Figure()

    # Add subtle quadrant background shading
    # Top-left
    fig.add_shape(
        type="rect",
        x0=x_min - 1, y0=0, x1=0, y1=y_max + 1,
        fillcolor="rgba(29, 185, 84, 0.03)",
        line=dict(width=0),
        layer="below",
    )
    # Top-right
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=x_max + 1, y1=y_max + 1,
        fillcolor="rgba(100, 100, 255, 0.03)",
        line=dict(width=0),
        layer="below",
    )
    # Bottom-left
    fig.add_shape(
        type="rect",
        x0=x_min - 1, y0=y_min - 1, x1=0, y1=0,
        fillcolor="rgba(255, 100, 100, 0.03)",
        line=dict(width=0),
        layer="below",
    )
    # Bottom-right
    fig.add_shape(
        type="rect",
        x0=0, y0=y_min - 1, x1=x_max + 1, y1=0,
        fillcolor="rgba(255, 200, 100, 0.03)",
        line=dict(width=0),
        layer="below",
    )

    # Add quadrant divider lines
    fig.add_shape(
        type="line",
        x0=0, y0=y_min - 1, x1=0, y1=y_max + 1,
        line=dict(color="#535353", width=1),
    )
    fig.add_shape(
        type="line",
        x0=x_min - 1, y0=0, x1=x_max + 1, y1=0,
        line=dict(color="#535353", width=1),
    )

    # Add quadrant labels as annotations
    label_offset = 0.3
    annotations = [
        dict(
            x=x_min + label_offset, y=y_max - label_offset,
            text=f"<b>{quadrant_labels['top_left']}</b>",
            showarrow=False, font=dict(size=14, color="#1DB954"),
            xanchor="left", yanchor="top",
        ),
        dict(
            x=x_max - label_offset, y=y_max - label_offset,
            text=f"<b>{quadrant_labels['top_right']}</b>",
            showarrow=False, font=dict(size=14, color="#6464ff"),
            xanchor="right", yanchor="top",
        ),
        dict(
            x=x_min + label_offset, y=y_min + label_offset,
            text=f"<b>{quadrant_labels['bottom_left']}</b>",
            showarrow=False, font=dict(size=14, color="#ff6464"),
            xanchor="left", yanchor="bottom",
        ),
        dict(
            x=x_max - label_offset, y=y_min + label_offset,
            text=f"<b>{quadrant_labels['bottom_right']}</b>",
            showarrow=False, font=dict(size=14, color="#ffc864"),
            xanchor="right", yanchor="bottom",
        ),
    ]

    # Calculate image sizes based on data range
    x_range = x_max - x_min
    y_range = y_max - y_min
    data_range = max(x_range, y_range, 1.0)
    # Base size relative to spread
    base_img_size = data_range * 0.06

    n_artists = len(artists)

    # First pass: collect positions and sizes using SCALED coordinates
    original_positions = []
    img_sizes = []
    for i, artist in enumerate(artists):
        # Use the already-scaled coordinates
        x, y = genre_x[i], genre_y[i]
        rank = rank_map[artist.id]

        # Size scales dramatically with preference (higher rank = much bigger)
        # Top artist is 3x size, bottom is 0.5x - very visible difference
        rank_fraction = rank / max(n_artists - 1, 1)  # 0 for top, 1 for bottom
        size_multiplier = 3.0 - (2.5 * rank_fraction)  # 3.0 -> 0.5
        img_size = base_img_size * size_multiplier

        original_positions.append((x, y))
        img_sizes.append(img_size)

    # Apply force-directed repulsion to reduce overlap
    adjusted_positions = _repel_overlapping_images(
        original_positions, img_sizes, iterations=100, repel_strength=0.5
    )

    # Second pass: add images at adjusted positions
    for i, artist in enumerate(artists):
        x, y = adjusted_positions[i]
        img_size = img_sizes[i]
        utility = posterior_summary[artist.id]["utility_mean"]
        rank = rank_map[artist.id]

        # Opacity scales dramatically: top artists fully visible, bottom fades out
        # Top 3: 100%, then fade from 80% down to 30% for worst
        rank_fraction = rank / max(n_artists - 1, 1)
        if rank < 3:
            opacity = 1.0
        else:
            opacity = 0.8 - (0.5 * rank_fraction)  # 0.8 -> 0.3

        is_top = rank == 0
        is_top3 = rank < 3

        # Text styling based on rank
        if is_top:
            text_color = "#1DB954"
        elif is_top3:
            text_color = "#ffffff"
        else:
            text_color = "#b3b3b3"

        # Add image if available - with circular masking effect
        if artist.image_url:
            # Draw dark background circle first (to mask square image corners)
            fig.add_shape(
                type="circle",
                x0=x - img_size / 2,
                y0=y - img_size / 2,
                x1=x + img_size / 2,
                y1=y + img_size / 2,
                xref="x",
                yref="y",
                line=dict(width=0),
                fillcolor="#121212",
                layer="above",
            )
            # Add image slightly smaller than circle so corners get clipped
            inner_size = img_size * 0.71  # sqrt(2)/2 to inscribe square in circle
            fig.add_layout_image(
                dict(
                    source=artist.image_url,
                    x=x,
                    y=y,
                    xref="x",
                    yref="y",
                    sizex=inner_size,
                    sizey=inner_size,
                    xanchor="center",
                    yanchor="middle",
                    layer="above",
                    opacity=opacity,
                )
            )
            # Add circular border on top
            border_color = "#1DB954" if is_top else ("#ffffff" if is_top3 else "#535353")
            border_width = 4 if is_top else (3 if is_top3 else 2)
            fig.add_shape(
                type="circle",
                x0=x - img_size / 2,
                y0=y - img_size / 2,
                x1=x + img_size / 2,
                y1=y + img_size / 2,
                xref="x",
                yref="y",
                line=dict(color=border_color, width=border_width),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )

        # Add text label and invisible marker for hover
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[artist.name],
                textposition="top center",
                textfont=dict(
                    size=13 if is_top else 11,
                    color=text_color,
                    family="Arial Black" if is_top else "Arial",
                ),
                marker=dict(
                    size=img_size * 100,  # Rough pixel conversion for hover area
                    color="rgba(0,0,0,0)",  # Invisible
                    line=dict(width=0),
                ),
                hovertemplate=(
                    f"<b>{artist.name}</b><br>"
                    f"Rank: #{rank + 1}<br>"
                    f"Utility: {utility:.2f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(
            text="Your Music Taste Map",
            font=dict(size=24, color="#ffffff"),
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[x_min - 0.5, x_max + 0.5],
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[y_min - 0.5, y_max + 0.5],
            scaleanchor="x",
            scaleratio=1,
        ),
        height=700,
        annotations=annotations,
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        hoverlabel=dict(
            bgcolor="#282828",
            font_size=14,
            font_color="#ffffff",
        ),
    )

    return fig


def plot_comparability_heatmap(
    artists: list[Artist],
    comparability_matrix: np.ndarray | Array,
) -> go.Figure:
    """Create heatmap of pairwise comparability probabilities.

    Args:
        artists: List of artists
        comparability_matrix: n_artists x n_artists matrix

    Returns:
        Plotly figure
    """
    names = [a.name for a in artists]

    # Convert JAX array to numpy if needed
    if hasattr(comparability_matrix, "to_py"):
        comparability_matrix = np.array(comparability_matrix)

    fig = go.Figure(
        data=go.Heatmap(
            z=comparability_matrix,
            x=names,
            y=names,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(title="P(comparable)"),
        )
    )

    fig.update_layout(
        title="Artist Comparability Matrix",
        height=max(500, len(artists) * 20),
        width=max(600, len(artists) * 20),
        xaxis=dict(tickangle=45),
    )

    return fig


def plot_comparison_history(
    comparisons_count: int,
    total_pairs: int,
) -> go.Figure:
    """Create a progress indicator for comparisons.

    Args:
        comparisons_count: Number of comparisons made
        total_pairs: Total possible pairs

    Returns:
        Plotly figure
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=comparisons_count,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Comparisons Made"},
            gauge={
                "axis": {"range": [0, total_pairs]},
                "bar": {"color": "steelblue"},
                "steps": [
                    {"range": [0, total_pairs * 0.3], "color": "lightgray"},
                    {"range": [total_pairs * 0.3, total_pairs * 0.7], "color": "gray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": min(comparisons_count + 10, total_pairs),
                },
            },
        )
    )

    fig.update_layout(height=300)

    return fig
