import os
import requests
from dotenv import load_dotenv
from typing import List, Dict

# === Configuration ===
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "azureblob-index")

# === Search Functionality ===
def search_articles(query: str, top_k: int = 3, min_score: float = 0.4) -> List[Dict]:
    """
    Search Azure Search index for articles matching the query.

    Args:
        query (str): Search query string.
        top_k (int): Maximum number of results to return (default: 3).
        min_score (float): Minimum relevance score for results (default: 0.4).

    Returns:
        List[Dict]: List of articles with title, url, and snippet fields.
    """
    if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX]):
        print("❌ Missing Azure Search configuration")
        raise ValueError("Missing Azure Search configuration")

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
        results = response.json().get("value", [])
    except requests.exceptions.RequestException as e:
        print(f"❌ Azure Search error: {e}")
        return []

    filtered = []
    for doc in results:
        content = doc.get("article_content") or doc.get("content")
        title = doc.get("title", "Untitled")
        url = doc.get("url") or "#"
        score = doc.get("@search.score", 0)

        if content and score >= min_score:
            snippet = content[:500] + "..." if len(content) > 500 else content
            filtered.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })

    print(f"🔍 Search query: {query}")
    print(f"✅ Articles returned: {len(filtered)}")
    return filtered
