
import os
import requests
from dotenv import load_dotenv

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "azureblob-index")

def search_articles(query, top_k=3, min_score=0.4):
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY or not AZURE_SEARCH_INDEX:
        raise ValueError("❌ Missing Azure Search configuration.")

    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2021-04-30-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_KEY
    }
    body = {
        "search": query,
        "top": top_k,
        "queryType": "simple",
        "searchMode": "any"
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Azure Search error: {e}")
        return []

    results = response.json().get("value", [])

    filtered = []
    for doc in results:
        content = doc.get("article_content") or doc.get("content")
        title = doc.get("title", "Untitled")
        url = doc.get("url") or "#"
        score = doc.get("@search.score", 0)

        if content and score >= min_score:
            filtered.append({
                "title": title,
                "url": url,
                "snippet": content[:500] + "..." if len(content) > 500 else content
            })

    print(f"🔍 Search query: {query}")
    print(f"📄 Articles returned: {len(filtered)}")
    return filtered
