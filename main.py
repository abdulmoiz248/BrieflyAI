import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")



model = SentenceTransformer("all-MiniLM-L6-v2")



def fetchHackerNews():
    r = requests.get("https://hn.algolia.com/api/v1/search_by_date?query=AI&tags=story").json()
    articles = []
    for h in r.get("hits", [])[:20]:
        title = h.get("title")
        url = h.get("url")  # use get() to avoid KeyError
        if title and url:
            articles.append({
                "title": title,
                "url": url,
                "source": "Hacker News",
                "published": h.get("created_at")
            })
    return articles


def deduplicateArticles(articles):
    titles = [a["title"] for a in articles]
    embeddings = model.encode(titles)
    keepIndexes = []
    for i, emb in enumerate(embeddings):
        isDuplicate = False
        for j in keepIndexes:
            sim = np.dot(emb, embeddings[j]) / (
                np.linalg.norm(emb) * np.linalg.norm(embeddings[j])
            )
            if sim > 0.85:
                isDuplicate = True
                break
        if not isDuplicate:
            keepIndexes.append(i)
    return [articles[i] for i in keepIndexes]

def scoreArticle(article):
    score = 0
    title = article["title"].lower()
    
    # Company keywords
    keywords = {
        "openai": 5, "gpt": 4, "chatgpt": 4,
        "google": 3, "gemini": 4,
        "microsoft": 3, "copilot": 3,
        "meta": 3, "llama": 3,
        "anthropic": 4, "claude": 4,
        "deepmind": 4, "gemini": 4,
        "announced": 3, "launched": 3, "released": 2,
        "new": 1, "update": 1, "partnership": 2, "acquisition": 2
    }
    
    for keyword, points in keywords.items():
        if keyword in title:
            score += points
    
    # Source authority (news sources, not research)
    reputable_sources = {
        "TechCrunch": 4, "The Verge": 4, "Wired": 3,
        "Forbes": 3, "Reuters": 4, "Associated Press": 4,
        "MIT Technology Review": 3, "CNBC": 3, "CNN": 2,
        "Hacker News": 2
    }
    
    for source, points in reputable_sources.items():
        if source in article["source"]:
            score += points
            break
    
    return score

def sendToDiscord(articles):
    """Send articles to Discord via webhook in batches of 5"""
    if not DISCORD_WEBHOOK_URL:
        print("Error: DISCORD_WEBHOOK_URL not set in environment")
        return
    
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    
    # Send in batches of 5
    for batch_idx in range(0, len(articles), 5):
        batch = articles[batch_idx:batch_idx + 5]
        
        # Build embed message
        embed = {
            "title": f"BrieflyAI — AI News ({timestamp})",
            "description": f"Batch {batch_idx // 5 + 1}",
            "color": 0x0066cc,
            "fields": []
        }
        
        for idx, a in enumerate(batch, start=batch_idx + 1):
            field_value = f"**Source:** {a['source']}\n**Link:** [{a['title']}]({a['url']})"
            embed["fields"].append({
                "name": f"{idx}. {a['title'][:256]}",
                "value": field_value,
                "inline": False
            })
        
        payload = {"embeds": [embed]}
        
        try:
            response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
            response.raise_for_status()
            print(f"Sent batch {batch_idx // 5 + 1} to Discord successfully")
        except requests.exceptions.RequestException as e:
            print(f"Error sending to Discord: {e}")

def run():
    articles = fetchHackerNews()

    articles = deduplicateArticles(articles)
    articles = sorted(articles, key=scoreArticle, reverse=True)[:20]

    sendToDiscord(articles)

run()
