from azure.cosmos import CosmosClient, PartitionKey, exceptions
import os
from datetime import datetime

def get_cosmos_client():
    """
    Initialize and return a Cosmos DB client.
    """
    endpoint = os.environ.get("COSMOS_DB_ENDPOINT")
    key = os.environ.get("COSMOS_DB_KEY")
    if not endpoint or not key:
        raise ValueError("Missing COSMOS_DB_ENDPOINT or COSMOS_DB_KEY environment variables")
    
    try:
        client = CosmosClient(endpoint, credential=key)
        print(f"🔗 Connected to Cosmos DB endpoint: {endpoint}")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Cosmos DB: {str(e)}")
        raise

def initialize_container(client, database_name, container_name):
    """
    Initialize or get a Cosmos DB container without setting throughput.
    """
    try:
        database = client.create_database_if_not_exists(id=database_name)
        print(f"📂 Using database: {database_name}")
        
        container = database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/user_id"),
            offer_throughput=None  # No throughput for serverless
        )
        print(f"📦 Using container: {container_name}")
        return container
    except exceptions.CosmosHttpResponseError as e:
        print(f"❌ Cosmos DB error: {str(e)}")
        raise

def update_preferences(user_id: str, fact_id: str, liked: bool):
    """
    Update or insert user preferences in Cosmos DB.
    
    Args:
        user_id (str): The user's ID (e.g., WhatsApp number).
        fact_id (str): The ID of the fact being liked or disliked.
        liked (bool): Whether the user liked (True) or disliked (False) the fact.
    """
    print(f"📥 update_preferences called with user_id={user_id}, fact_id={fact_id}, liked={liked}")
    
    try:
        client = get_cosmos_client()
        container = initialize_container(
            client,
            os.environ.get("COSMOS_DB_DATABASE", "fique_db"),
            os.environ.get("COSMOS_DB_CONTAINER", "preferences")
        )
        
        item = {
            "id": f"{user_id}_{fact_id}",
            "user_id": user_id,
            "fact_id": fact_id,
            "liked": liked,
            "timestamp": int(datetime.utcnow().timestamp())
        }
        
        container.upsert_item(item)
        print(f"✅ Successfully upserted preference for user_id={user_id}, fact_id={fact_id}, liked={liked}")
        
    except Exception as e:
        print(f"❌ Error in update_preferences: {str(e)}")
        raise
