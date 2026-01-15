# Artist Preference Ranking via Pairwise Song Comparison

## Project Characterization

### Overview

Infer partial preference orderings over musical artists from pairwise song comparisons, where users may abstain when artists occupy incomparable musical spaces. The model treats genre as a continuous latent manifold rather than discrete categories, with comparability probability decaying as a function of distance in this space.

---

## Data Regime

### Scale Parameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| Artists | ~25 | Target inference population |
| Songs per artist | 1–10 | Variable catalog depth |
| Total comparisons | 50–200 | User fatigue constraint |
| Comparisons per artist | ~2–8 | Sparse; implies need for strong priors |
| Users | 1 | Single-user preference model (initially) |

### Observation Structure

Each query presents two songs $(x, y)$ from distinct artists $(A, B)$.

Response space:
$$R \in \{x \succ y,\; y \succ x,\; \bot\}$$

where $\bot$ (abstain) indicates **categorical incomparability**—the user judges the artists as "doing fundamentally different things" rather than expressing uncertainty about relative quality.

**Song-level treatment:** Multiple comparisons between the same artist pair (different songs) are treated as independent observations. Each comparison is an i.i.d. draw given the latent artist utilities and song-level noise.

### Sparsity Implications

With ~300 possible artist pairs and 50–200 observations:
- Most pairs unobserved
- Transitivity and latent structure must propagate information
- Model complexity bounded to ~30–50 effective parameters

---

## Available Metadata Priors

### Per-Artist Features

| Feature | Type | Source | Quality |
|---------|------|--------|---------|
| Genre tags | Categorical / multi-label | Spotify API, MusicBrainz | Noisy but informative |
| Audio embeddings | Dense vector | Spotify audio features, Essentia | High-quality continuous |
| Popularity metrics | Scalar | Streaming counts | Auxiliary |
| Release year / era | Temporal | Catalog metadata | Contextual |

### Prior Construction via LLM

Query an LLM for pairwise artist distances on a 1–5 scale (leveraging its knowledge of artist names and musical relationships). Fit a K-dimensional MDS projection to obtain prior mean positions:

$$\mu_{\text{prior}}^{(a)} = \text{MDS}_K(\{d_{ab}^{\text{LLM}}\})$$

This provides an informed initialization that respects known genre relationships while allowing the model to refine positions based on observed comparisons.

**Fallback (no metadata):** Standard normal prior with anchor constraints.

---

## Modeling Approach

### Core Assumption

Genre structure is a **continuous low-dimensional manifold** (1–3 dimensions), not a set of discrete clusters. Artists spanning genre boundaries occupy intermediate positions naturally. Comparability decays smoothly with distance in this space.

### Latent Structure

**Genre position:**
$$g_a \in \mathbb{R}^K, \quad K \in \{1, 2, 3\}$$

**Artist preference utility:**
$$s_a \in \mathbb{R}$$

**Artist consistency** (inverse variance of song quality):
$$\lambda_a > 0$$

Artists with high $\lambda_a$ have consistent song quality; low $\lambda_a$ indicates a "hit-or-miss" catalog.

### Song-Level Model

For a song $t$ by artist $a$:
$$U_{a,t} = s_a + \epsilon_{a,t}, \quad \epsilon_{a,t} \sim \mathcal{N}(0, \lambda_a^{-1})$$

When comparing songs $(t, t')$ from artists $(A, B)$:
$$U_{A,t} - U_{B,t'} \sim \mathcal{N}(s_A - s_B, \; \lambda_A^{-1} + \lambda_B^{-1})$$

### Hierarchical Observation Model

**Stage 1 — Comparability (function of genre distance):**

For $K = 1$:
$$P(\text{comparable} \mid A, B) = \sigma\left(\alpha - \frac{(g_A - g_B)^2}{\tau^2}\right)$$

For $K \geq 2$ (ARD lengthscales):
$$P(\text{comparable} \mid A, B) = \sigma\left(\alpha - \sum_{k=1}^{K} \frac{(g_{A,k} - g_{B,k})^2}{\tau_k^2}\right)$$

- $\tau_k$ controls the "comparable radius" along dimension $k$
- $\alpha$ sets baseline comparability at zero distance
- ARD structure learns which genre dimensions matter most for comparability

**Stage 2 — Direction (conditional on comparability):**

$$P(x \succ y \mid \text{comparable}) = \Phi\left(\frac{s_A - s_B}{\sqrt{\lambda_A^{-1} + \lambda_B^{-1}}}\right)$$

where $\Phi$ is the standard normal CDF (probit link, natural given Gaussian song noise).

**Joint likelihood for observation $r$ on pair $(A, B)$ with songs $(t, t')$:**

$$P(r = \bot) = 1 - P(\text{comparable})$$

$$P(r = A \succ B) = P(\text{comparable}) \cdot \Phi\left(\frac{s_A - s_B}{\sqrt{\lambda_A^{-1} + \lambda_B^{-1}}}\right)$$

$$P(r = B \succ A) = P(\text{comparable}) \cdot \Phi\left(\frac{s_B - s_A}{\sqrt{\lambda_A^{-1} + \lambda_B^{-1}}}\right)$$

### Scale Identifiability

The model fixes scales to ensure identifiability:

| Parameter | Convention | Rationale |
|-----------|------------|-----------|
| Genre prior scale | $\sigma_g = 1$ | Makes $\tau_k$ interpretable as "SDs in genre space" |
| Utility scale | Learned $\sigma_s$ | Probit link has implicit scale; $\sigma_s$ captures population spread |
| Song noise | Per-artist $\lambda_a$ | Explicitly models artist consistency |

### Priors

**Genre positions:**
$$g_a \sim \mathcal{N}(\mu_{\text{prior}}^{(a)}, 1)$$

**Artist utilities:**
$$s_a \sim \mathcal{N}(0, \sigma_s^2), \quad \sigma_s \sim \text{Half-Normal}(1)$$

**Artist consistency:**
$$\lambda_a \sim \text{Gamma}(2, 1)$$

Prior mean of 2 (moderate consistency), with support for both very consistent ($\lambda_a > 5$) and inconsistent ($\lambda_a < 1$) artists.

**Comparability lengthscales:**
$$\tau_k \sim \text{Half-Normal}(1)$$

**Comparability intercept:**
$$\alpha \sim \mathcal{N}(1, 1)$$

Prior centered at $\alpha = 1$ implies ~73% comparability at zero genre distance.

---

## Output Specification

### Primary Output

### Primary Output

**Posterior distributions** over:
- Artist utilities $\{s_a\}$
- Genre positions $\{g_a\}$
- Artist consistency $\{\lambda_a\}$
- Lengthscales $\{\tau_k\}$

### Visualization

Display artists as points in the learned genre space (1D or 2D projection), with vertical position or color indicating **local preference score** (posterior mean $\hat{s}_a$). 

For $K = 1$: Scatter plot with genre position on x-axis, utility on y-axis.

For $K = 2$: 2D genre map with color/size encoding utility.

Optionally overlay uncertainty (error bars or transparency from posterior SD).

---

## Active Comparison Selection

Select pairs to maximize expected information gain:

$$\text{score}(A, B) = P(\text{comparable} \mid A, B) \times H(R \mid A, B)$$

where $H(R \mid A, B)$ is the entropy of the three-outcome response distribution (prefer A, prefer B, abstain).

This single mechanism naturally balances:
- **Answerability**: low comparability → low score (don't waste user time)
- **Informativeness**: low uncertainty → low score (we already know the answer)

No phasing or threshold tuning required. At each step, compute scores for all unobserved pairs and sample the highest-scoring candidate (or top-k with some randomization to avoid deterministic loops).

## Dimensionality Selection

| $K$ | Structure Captured | Parameters | Min. Comparisons | Recommendation |
|-----|-------------------|------------|------------------|----------------|
| 1 | Single axis (e.g., electronic ↔ acoustic) | 1 lengthscale | ~75–100 | Default; start here |
| 2 | Two axes (e.g., energy × mood) | 2 lengthscales (ARD) | ~150–200 | If 1D residuals show misfit |
| 3 | Full genre space | 3 lengthscales (ARD) | ~300+ | Outside current data budget |

Model selection via WAIC/LOO-CV or posterior predictive checks on held-out comparisons.

---


### Identifiability Constraints

- Fix $g_{a_0} = 0$ (genre position anchor)
- Fix $s_{a_0} = 0$ (utility anchor)
- For $K > 1$: fix rotation (e.g., $g_{a_1,2} = 0$, placing $a_1$ on first axis)
