"""Tests for visualization functions - especially image layout."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.data import Artist
from src.viz import plot_genre_quadrants, plot_utility_ranking


def create_mock_artists(n: int = 10, with_images: bool = True) -> list[Artist]:
    """Create mock artists for testing."""
    artists = []
    for i in range(n):
        artists.append(Artist(
            id=f"artist_{i}",
            name=f"Artist {i}",
            genres=["rock", "indie"],
            popularity=50 + i * 3,
            image_url=f"https://example.com/img_{i}.jpg" if with_images else None,
        ))
    return artists


def create_mock_posterior(artists: list[Artist], spread: float = 1.0) -> dict:
    """Create mock posterior with controllable spread."""
    np.random.seed(42)  # Reproducible
    posterior = {}
    n = len(artists)

    for i, artist in enumerate(artists):
        # Create positions that spread artists out
        # Use golden ratio spiral for good distribution
        angle = i * 2.4  # Golden angle
        radius = spread * np.sqrt(i + 1) / np.sqrt(n)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        posterior[artist.id] = {
            "utility_mean": 1.0 - (i / n),  # Rank by index
            "utility_std": 0.1,
            "genre_mean": [x, y],
            "genre_std": [0.1, 0.1],
            "lambda_mean": 1.0,
            "lambda_std": 0.1,
        }

    return posterior


def create_clustered_posterior(artists: list[Artist], cluster_size: float = 0.1) -> dict:
    """Create posterior where artists are clustered (worst case for overlap)."""
    np.random.seed(42)
    posterior = {}
    n = len(artists)

    for i, artist in enumerate(artists):
        # All artists clustered near origin
        x = np.random.uniform(-cluster_size, cluster_size)
        y = np.random.uniform(-cluster_size, cluster_size)

        posterior[artist.id] = {
            "utility_mean": 1.0 - (i / n),
            "utility_std": 0.1,
            "genre_mean": [x, y],
            "genre_std": [0.1, 0.1],
            "lambda_mean": 1.0,
            "lambda_std": 0.1,
        }

    return posterior


class TestQuadrantVisualization:
    """Test the quadrant visualization."""

    def test_creates_figure(self):
        """Basic test that figure is created."""
        artists = create_mock_artists(5)
        posterior = create_mock_posterior(artists)

        fig = plot_genre_quadrants(artists, posterior)

        assert fig is not None
        assert hasattr(fig, 'data')

    def test_with_quadrant_labels(self):
        """Test with custom quadrant labels."""
        artists = create_mock_artists(5)
        posterior = create_mock_posterior(artists)
        labels = {
            "top_left": "Indie Dreams",
            "top_right": "Electronic Pulse",
            "bottom_left": "Raw Energy",
            "bottom_right": "Smooth Grooves",
        }

        fig = plot_genre_quadrants(artists, posterior, quadrant_labels=labels)

        assert fig is not None

    def test_spread_artists_no_overlap(self):
        """Test that well-spread artists don't overlap."""
        artists = create_mock_artists(10)
        posterior = create_mock_posterior(artists, spread=2.0)

        fig = plot_genre_quadrants(artists, posterior)

        # Check that layout images were added
        assert len(fig.layout.images) == 10

        # Check image sizes are reasonable (positive and not huge)
        for img in fig.layout.images:
            assert img.sizex > 0
            assert img.sizex < 10.0  # Sanity check

    def test_clustered_artists_overlap_handling(self):
        """Test handling of clustered/overlapping artists."""
        artists = create_mock_artists(10)
        posterior = create_clustered_posterior(artists, cluster_size=0.05)

        fig = plot_genre_quadrants(artists, posterior)

        # Figure should still be created
        assert fig is not None

        # Images should be smaller when clustered
        # This is a design goal - reduce overlap
        for img in fig.layout.images:
            assert img.sizex > 0

    def test_without_images(self):
        """Test visualization when artists have no images."""
        artists = create_mock_artists(5, with_images=False)
        posterior = create_mock_posterior(artists)

        fig = plot_genre_quadrants(artists, posterior)

        assert fig is not None
        # No layout images should be added
        assert len(fig.layout.images) == 0

    def test_image_size_scales_with_rank(self):
        """Test that higher-ranked artists have larger images."""
        artists = create_mock_artists(5)
        posterior = create_mock_posterior(artists, spread=2.0)

        fig = plot_genre_quadrants(artists, posterior)

        # First image (rank 0) should be larger than last
        sizes = [img.sizex for img in fig.layout.images]
        assert sizes[0] > sizes[-1], "Top artist should have larger image"


class TestUtilityRanking:
    """Test the utility ranking bar chart."""

    def test_creates_figure(self):
        """Basic test that figure is created."""
        artists = create_mock_artists(5)
        posterior = create_mock_posterior(artists)

        fig = plot_utility_ranking(artists, posterior)

        assert fig is not None

    def test_sorted_by_utility(self):
        """Test that artists are sorted by utility."""
        artists = create_mock_artists(5)
        posterior = create_mock_posterior(artists)

        fig = plot_utility_ranking(artists, posterior)

        # Get y-axis labels (artist names)
        y_data = fig.data[0].y

        # First should be highest utility
        assert y_data[0] == "Artist 0"  # Highest utility in our mock


def calculate_overlap_score(positions: list[tuple[float, float]], sizes: list[float]) -> float:
    """Calculate overlap score - lower is better."""
    n = len(positions)
    if n < 2:
        return 0.0

    overlap_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            s1, s2 = sizes[i], sizes[j]

            # Distance between centers
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Overlap if distance < sum of radii
            if dist < (s1 + s2) / 2:
                overlap_count += 1

    # Normalize by total possible pairs
    total_pairs = n * (n - 1) / 2
    return overlap_count / total_pairs


class TestOverlapMetrics:
    """Test overlap handling."""

    def test_overlap_score_no_overlap(self):
        """Test overlap score with well-separated points."""
        positions = [(0, 0), (2, 0), (0, 2), (2, 2)]
        sizes = [0.5, 0.5, 0.5, 0.5]

        score = calculate_overlap_score(positions, sizes)
        assert score == 0.0

    def test_overlap_score_full_overlap(self):
        """Test overlap score with all points at same location."""
        positions = [(0, 0), (0, 0), (0, 0)]
        sizes = [0.5, 0.5, 0.5]

        score = calculate_overlap_score(positions, sizes)
        assert score == 1.0

    def test_quadrant_viz_overlap_rate(self):
        """Test that quadrant viz achieves reasonable overlap rate."""
        # Run multiple random scenarios
        overlap_scores = []

        for seed in range(5):
            np.random.seed(seed)
            artists = create_mock_artists(15)

            # Create somewhat clustered data (realistic scenario)
            posterior = {}
            for i, artist in enumerate(artists):
                x = np.random.normal(0, 0.3)
                y = np.random.normal(0, 0.3)
                posterior[artist.id] = {
                    "utility_mean": np.random.uniform(-1, 1),
                    "utility_std": 0.1,
                    "genre_mean": [x, y],
                    "genre_std": [0.1, 0.1],
                    "lambda_mean": 1.0,
                    "lambda_std": 0.1,
                }

            fig = plot_genre_quadrants(artists, posterior)

            # Extract positions and sizes from figure
            positions = []
            sizes = []
            for img in fig.layout.images:
                positions.append((img.x, img.y))
                sizes.append(img.sizex)

            if positions:
                score = calculate_overlap_score(positions, sizes)
                overlap_scores.append(score)

        # Average overlap should be reasonable (less than 50%)
        avg_overlap = np.mean(overlap_scores)
        print(f"Average overlap score: {avg_overlap:.2%}")

        # With repulsion algorithm, we expect <15% overlap
        assert avg_overlap < 0.15, f"Overlap rate {avg_overlap:.1%} exceeds 15% threshold"

    def test_extreme_clustering_overlap(self):
        """Test overlap handling with extreme clustering (all artists at same point)."""
        overlap_scores = []

        for seed in range(20):  # Many trials for robustness
            np.random.seed(seed)
            artists = create_mock_artists(20)  # More artists = harder problem

            # All artists at nearly the same point - worst case
            posterior = {}
            for i, artist in enumerate(artists):
                x = np.random.normal(0, 0.05)  # Very tight cluster
                y = np.random.normal(0, 0.05)
                posterior[artist.id] = {
                    "utility_mean": np.random.uniform(-1, 1),
                    "utility_std": 0.1,
                    "genre_mean": [x, y],
                    "genre_std": [0.1, 0.1],
                    "lambda_mean": 1.0,
                    "lambda_std": 0.1,
                }

            fig = plot_genre_quadrants(artists, posterior)

            positions = []
            sizes = []
            for img in fig.layout.images:
                positions.append((img.x, img.y))
                sizes.append(img.sizex)

            if positions:
                score = calculate_overlap_score(positions, sizes)
                overlap_scores.append(score)

        avg_overlap = np.mean(overlap_scores)
        max_overlap = max(overlap_scores)
        print(f"Extreme clustering - Avg: {avg_overlap:.2%}, Max: {max_overlap:.2%}")

        # Even with extreme clustering, repulsion should keep overlap manageable
        assert avg_overlap < 0.25, f"Extreme clustering avg overlap {avg_overlap:.1%} exceeds 25%"
