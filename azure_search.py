import os
import requests
from dotenv import load_dotenv

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "azureblob-index")

def search_articles(query, top_k=3):
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY or not AZURE_SEARCH_INDEX:
        raise ValueError("❌ Missing Azure Search configuration.")

    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2021-04-30-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_KEY
    }
    body = {
        "search": query,
        "top": top_k
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()

    results = response.json().get("value", [])
    return [f"{doc['title']}\n{doc['article_content']}" for doc in results if 'article_content' in doc]
