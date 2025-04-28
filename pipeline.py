
import os, hashlib, datetime, feedparser, requests, re
from bs4 import BeautifulSoup

# ---------- CONFIG ----------
NEWS_SOURCES = [
    "https://www.dw.com/en/rss",
    "http://feeds.bbci.co.uk/news/world/rss.xml"
    # add more RSS URLs here
]
MAX_ITEMS_PER_FEED = 5  # to keep runtime short
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")  # set in GitHub Secrets
# ---------- /CONFIG ----------

GOOGLE_API_URL = "https://translation.googleapis.com/language/translate/v2"

def translate_to_ar(text: str) -> str:
    """Translate English text to Arabic via Google Translate API.
    Falls back to original text if no key is provided."""
    if not GOOGLE_API_KEY:
        return text
    resp = requests.post(
        GOOGLE_API_URL,
        params={"key": GOOGLE_API_KEY},
        json={"q": text, "target": "ar"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["data"]["translations"][0]["translatedText"]

def textrank_summary(text: str, max_sentences: int = 3) -> str:
    """Very lightweight TextRank using sumy."""
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = TextRankSummarizer()(parser.document, max_sentences)
    return " ".join(str(sentence) for sentence in summary)

def slugify(value: str) -> str:
    value = re.sub(r"[\s_]+", "-", value.strip())
    return re.sub(r"[^\w\-]", "", value).lower()[:80]

def save_markdown(title_ar: str, summary_ar: str, source_url: str):
    slug = slugify(title_ar)
    uid_path = f"content/{slug}.md"
    if os.path.exists(uid_path):
        return
    date_iso = datetime.datetime.utcnow().isoformat()
    md = f"""---
title: "{title_ar}"
date: {date_iso}
source: "{source_url}"
slug: "{slug}"
---

{summary_ar}

[المصدر الأصلي]({source_url})
"""
    with open(uid_path, "w", encoding="utf-8") as f:
        f.write(md)

def fetch_full_text(url: str) -> str:
    try:
        html = requests.get(url, timeout=15).text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        return paragraphs
    except Exception:
        return ""

def pipeline():
    for feed_url in NEWS_SOURCES:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:MAX_ITEMS_PER_FEED]:
            article_url = entry.link
            article_text = fetch_full_text(article_url)
            if len(article_text) < 300:
                continue
            summary_en = textrank_summary(article_text)
            summary_ar = translate_to_ar(summary_en)
            title_ar = translate_to_ar(entry.title)
            save_markdown(title_ar, summary_ar, article_url)

if __name__ == "__main__":
    pipeline()
