# Hosting Cost Analysis

## Scenario
- 20 friends using the app
- Each user copy-pastes their Spotify JSON data
- You pay for Claude API costs (LLM priors generation)

---

## Cost Breakdown

### 1. Streamlit Cloud (Free Tier) - Recommended

**Hosting: $0/month**
- Free for public apps
- Free for private apps with up to 1 viewer at a time
- Your friends can use it one at a time for free

**Limitations:**
- Apps sleep after inactivity
- 1GB RAM limit (plenty for this app)
- CPU-only (JAX inference still works, just slower)

**Setup:**
1. Push to GitHub
2. Connect to streamlit.io
3. Deploy from GitHub repo

---

### 2. Claude API Costs (LLM Priors)

**Per-user cost estimate:**

The LLM priors call sends ~25 artist names and asks for pairwise genre distances.

- **Input tokens**: ~500-800 (artist list + prompt)
- **Output tokens**: ~1,500-2,500 (distance matrix JSON)
- **Model**: Claude Sonnet 4

**Pricing (Sonnet 4):**
- Input: $3 / million tokens
- Output: $15 / million tokens

**Cost per user:**
- Input: 800 tokens × $0.000003 = $0.0024
- Output: 2,000 tokens × $0.000015 = $0.030
- **Total: ~$0.03 per user**

**For 20 friends: ~$0.60 total** (one-time per session)

---

### 3. Alternative Hosting Options

| Option | Cost | Notes |
|--------|------|-------|
| **Streamlit Cloud** | Free | Best option, easy setup |
| **Railway.app** | ~$5/mo | Free tier available, auto-sleep |
| **Render.com** | ~$7/mo | Free tier, 750 hrs/mo |
| **Fly.io** | ~$5/mo | Free allowance, scales to zero |
| **Your own VPS** | ~$5/mo | DigitalOcean/Linode droplet |
| **Vercel** | Free | Not ideal for Streamlit |

---

## Recommended Setup

### Option A: Streamlit Cloud (Simplest)

1. Create GitHub repo (can be private)
2. Go to share.streamlit.io
3. Deploy from repo
4. Share link with friends

**Pros:**
- Zero hosting cost
- Automatic deploys
- HTTPS included
- Custom subdomain

**Cons:**
- Apps sleep after ~7 days inactivity
- Cold starts take 30-60 seconds
- 1 concurrent viewer on free tier

### Option B: Railway/Render (Always-On)

1. Connect GitHub repo
2. Set environment variables:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Deploy

**Pros:**
- No cold starts
- Multiple concurrent users
- More RAM/CPU

**Cons:**
- $5-10/month

---

## Environment Variables Needed

```bash
# Required for LLM priors
ANTHROPIC_API_KEY=sk-ant-api03-...

# Optional: Spotify OAuth (if you get credentials)
SPOTIPY_CLIENT_ID=...
SPOTIPY_CLIENT_SECRET=...
SPOTIPY_REDIRECT_URI=https://your-app.streamlit.app
```

---

## Total Monthly Cost Estimate

| Item | Cost |
|------|------|
| Hosting (Streamlit Cloud) | $0 |
| Claude API (20 users, one-time) | ~$0.60 |
| **Total** | **~$0.60** |

If friends use it multiple times per month, budget ~$3-5/month for API costs.

---

## Sharing With Friends

### Option 1: Direct Link
Just share the Streamlit Cloud URL. Friends:
1. Go to link
2. Paste their Spotify JSON
3. Start ranking

### Option 2: Password Protected
Add simple password in Streamlit:
```python
if st.text_input("Password", type="password") != "your-password":
    st.stop()
```

### Option 3: Invite-Only via GitHub
Make repo private, add friends as collaborators, they can access the app.

---

## JSON Data Instructions for Friends

Share this with your friends:

> **How to get your Spotify data:**
>
> 1. Go to [Spotify Web API Console](https://developer.spotify.com/console/get-current-user-top-artists-and-tracks/)
> 2. Click "Get Token" and authorize
> 3. For Artists: Set `type=artists`, `limit=50`
> 4. For Tracks: Set `type=tracks`, `limit=50` (do this 5 times with offset 0, 50, 100, 150, 200)
> 5. Copy each JSON response
> 6. Paste into the app

Or use the curl commands:
```bash
# Get token from https://developer.spotify.com/console/
TOKEN="your-token"

# Top Artists
curl -X GET "https://api.spotify.com/v1/me/top/artists?limit=50" \
  -H "Authorization: Bearer $TOKEN" > top_artists.json

# Top Tracks (multiple pages)
curl -X GET "https://api.spotify.com/v1/me/top/tracks?limit=50&offset=0" \
  -H "Authorization: Bearer $TOKEN" > top_tracks_1.json
```
