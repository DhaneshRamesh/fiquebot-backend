from azure.cosmos import CosmosClient, PartitionKey, exceptions
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

COSMOS_DB_ENDPOINT = os.environ.get("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.environ.get("COSMOS_DB_KEY")
DATABASE_NAME = os.environ.get("COSMOS_DB_DATABASE", "fique_db")
CONTAINER_NAME = os.environ.get("COSMOS_DB_CONTAINER", "preferences")

# Validate environment variables
if not COSMOS_DB_ENDPOINT or not COSMOS_DB_KEY:
    raise EnvironmentError("Missing COSMOS_DB_ENDPOINT or COSMOS_DB_KEY environment variables")

def update_preferences(user_id: str, fact_id: str, liked: bool) -> None:
    """
    Update or insert a user's preference for a fact in Cosmos DB.

    Args:
        user_id (str): The user's ID (e.g., WhatsApp number or UUID).
        fact_id (str): The ID of the fact being liked or disliked.
        liked (bool): Whether the user liked (True) or disliked (False) the fact.

    Raises:
        exceptions.CosmosHttpResponseError: If there's an error interacting with Cosmos DB.
        Exception: For other unexpected errors.
    """
    try:
        # Initialize Cosmos client
        client = CosmosClient(COSMOS_DB_ENDPOINT, credential=COSMOS_DB_KEY)

        # Get or create database
        database = client.create_database_if_not_exists(id=DATABASE_NAME)

        # Define container properties with partition key
        container_definition = {
            "id": CONTAINER_NAME,
            "partition_key": PartitionKey(path="/user_id")
        }

        # Get or create container
        container = database.create_container_if_not_exists(
            id=CONTAINER_NAME,
            partition_key=container_definition["partition_key"],
            offer_throughput=400  # Adjust throughput as needed
        )

        # Prepare item to upsert
        preference_item = {
            "id": f"{user_id}_{fact_id}",  # Unique ID for the preference
            "user_id": user_id,  # Partition key
            "fact_id": fact_id,
            "liked": liked,
            "timestamp": int(os.times().elapsed)  # Store timestamp for record
        }

        # Upsert the preference (creates new or updates existing)
        container.upsert_item(preference_item)
        print(f"✅ Updated preference for user_id={user_id}, fact_id={fact_id}, liked={liked}")

    except exceptions.CosmosHttpResponseError as e:
        print(f"❌ Cosmos DB error: {str(e)}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error in update_preferences: {str(e)}")
        raise
    finally:
        client.close()
