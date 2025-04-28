
# Auto Arabic News Prototype

This repo holds a minimal prototype that:
1. Pulls articles from RSS feeds.
2. Generates Arabic summaries (≤3 sentences).
3. Saves each summary as a Markdown post.
4. Commits & pushes via GitHub Actions every hour.

## Quick start

1. Fork or clone this repo to your GitHub account.
2. Edit `pipeline.py` → add your favourite RSS feeds.
3. In **Repo Settings ➜ Secrets** add `GOOGLE_API_KEY` (or leave empty to skip translation).
4. Enable **GitHub Pages** → deploy from `main` branch (root).
5. Wait for the first Action run – your site will appear at  
   `https://<username>.github.io/<repo>/`.

## Local test
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py
```

All generated posts land in the `content/` folder.
