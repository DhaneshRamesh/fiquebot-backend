from azure.cosmos import CosmosClient
import os
import uuid
import time

def update_preferences(user_id: str, fact_id: str, liked: bool, confidence: float = 1.0) -> None:
    """
    Update user preferences in Cosmos DB with liking information.

    Args:
        user_id (str): The user's ID (e.g., WhatsApp number).
        fact_id (str): The ID of the fact being liked or disliked.
        liked (bool): Whether the user liked (True) or disliked (False) the fact.
        confidence (float): Confidence score for implicit liking (default 1.0 for explicit).
    """
    try:
        client = CosmosClient(os.environ["COSMOS_DB_ENDPOINT"], os.environ["COSMOS_DB_KEY"])
        database = client.get_database_client("FiqueDB")
        container = database.get_container_client("Preferences")
        
        # Generate a unique ID for the preference entry
        preference_id = f"{user_id}_{fact_id}_{str(uuid.uuid4())}"
        
        # Create or update preference entry
        preference = {
            "id": preference_id,
            "user_id": user_id,
            "fact_id": fact_id,
            "liked": liked,
            "confidence": confidence,
            "timestamp": int(time.time())
        }
        container.upsert_item(preference)
        print(f"✅ Updated preference for user {user_id}: fact_id={fact_id}, liked={liked}, confidence={confidence}")
    except Exception as e:
        print(f"❌ Error updating preferences: {str(e)}")
        raise
