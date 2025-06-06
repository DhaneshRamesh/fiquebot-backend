from azure.cosmos import CosmosClient
from dotenv import load_dotenv
import os

load_dotenv()

# Cosmos DB configuration
ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
KEY = os.getenv("COSMOS_DB_KEY")
DATABASE_NAME = "fiquebot"
CONTAINER_NAME = "user_profiles"

# Initialize client
client = CosmosClient(ENDPOINT, KEY)
database = client.get_database_client(DATABASE_NAME)
container = database.get_container_client(CONTAINER_NAME)

def get_user_profile(user_id: str) -> dict:
    try:
        item = container.read_item(item=user_id, partition_key=user_id)
        return item
    except:
        # Create new profile if not found
        profile = {
            "id": user_id,
            "user_id": user_id,
            "liked_facts": [],
            "disliked_facts": []
        }
        container.upsert_item(profile)
        return profile

def update_preferences(user_id: str, fact_id: str, liked: bool) -> None:
    profile = get_user_profile(user_id)
    if liked:
        if fact_id not in profile["liked_facts"]:
            profile["liked_facts"].append(fact_id)
        if fact_id in profile["disliked_facts"]:
            profile["disliked_facts"].remove(fact_id)
    else:
        if fact_id not in profile["disliked_facts"]:
            profile["disliked_facts"].append(fact_id)
        if fact_id in profile["liked_facts"]:
            profile["liked_facts"].remove(fact_id)
    container.upsert_item(profile)
