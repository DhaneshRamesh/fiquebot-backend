from azure.cosmos import CosmosClient, PartitionKey
import os

def get_cosmos_client():
    """
    Initialize Cosmos DB client and return the user_profiles container.
    Uses environment variables COSMOS_DB_ENDPOINT and COSMOS_DB_KEY from Render.
    Database: fiquebot, Container: user_profiles, Partition Key: /user_id
    """
    endpoint = os.environ.get("COSMOS_DB_ENDPOINT")
    key = os.environ.get("COSMOS_DB_KEY")
    if not endpoint or not key:
        raise EnvironmentError("COSMOS_DB_ENDPOINT or COSMOS_DB_KEY not set")
    client = CosmosClient(endpoint, credential=key)
    database = client.get_database_client("fiquebot")
    container = database.get_container_client("user_profiles")
    return container

def get_user_profile(user_id: str):
    """
    Retrieve user profile from Cosmos DB by user_id.
    If no profile exists, return a default profile with empty liked/disliked facts.
    Args:
        user_id (str): User ID (e.g., 'whatsapp:+447123456789' or 'uuid-...').
    Returns:
        dict: User profile with id, user_id, liked_facts, disliked_facts.
    """
    container = get_cosmos_client()
    try:
        profile = container.read_item(item=user_id, partition_key=user_id)
        return profile
    except:
        # If profile doesn't exist, return default
        return {
            "id": user_id,
            "user_id": user_id,
            "liked_facts": [],
            "disliked_facts": []
        }

def update_preferences(user_id: str, fact_id: str, liked: bool):
    """
    Update user preferences in Cosmos DB by adding fact_id to liked_facts or disliked_facts.
    Ensures no duplicate or conflicting fact IDs.
    Args:
        user_id (str): User ID (e.g., 'whatsapp:+447123456789' or 'uuid-...').
        fact_id (str): Fact ID to like/dislike (e.g., 'fact001').
        liked (bool): True to add to liked_facts, False to add to disliked_facts.
    """
    container = get_cosmos_client()
    profile = get_user_profile(user_id)
    
    # Update liked/disliked facts
    liked_facts = profile.get("liked_facts", [])
    disliked_facts = profile.get("disliked_facts", [])
    
    # Remove fact_id from both lists to avoid conflicts
    if fact_id in liked_facts:
        liked_facts.remove(fact_id)
    if fact_id in disliked_facts:
        disliked_facts.remove(fact_id)
    
    # Add to appropriate list
    if liked:
        liked_facts.append(fact_id)
    else:
        disliked_facts.append(fact_id)
    
    # Update profile
    profile["liked_facts"] = liked_facts
    profile["disliked_facts"] = disliked_facts
    
    # Upsert to Cosmos DB
    container.upsert_item({
        "id": user_id,
        "user_id": user_id,
        "liked_facts": liked_facts,
        "disliked_facts": disliked_facts
    })
