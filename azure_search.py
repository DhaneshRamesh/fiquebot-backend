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
        content = doc.get('article_content')  # Make sure this matches your index field
        title = doc.get('title', 'Untitled')
        score = doc.get('@search.score', 0)

        if content and score >= min_score:
            filtered.append(f"{title}\n{content}")

    # 🔍 Debug logs
    print(f"🔍 Search query: {query}")
    print(f"📄 Matches found: {len(filtered)}")
    for i, doc in enumerate(filtered):
        print(f"📝 [{i+1}] {doc[:200]}...\n")

    return filtered
