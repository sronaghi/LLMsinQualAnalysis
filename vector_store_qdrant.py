from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import requests
import json

def get_embedding(text: str) -> list:
    """Get embedding for text using Stanford's API."""
    API_ENDPOINT = "https://apim.stanfordhealthcare.org/openai3/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"
    API_KEY = # key
    
    headers = {
        "Ocp-Apim-Subscription-Key": API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"input": text}
    
    response = requests.post(API_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    
    return response.json()["data"][0]["embedding"]

def test_qdrant_queries():
    # Initialize client
    client = QdrantClient(path="./qdrant_db")
    collection_name = "transcribed_interviews"

    print("\n1. Testing vector search for 'diabetes':")
    # Get embedding for 'diabetes'
    query_vector = get_embedding("diabetes")
    
    # Search without filters first
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
        with_payload=True,
    )

    print(f"\nFound {len(results)} results without filters")
    if results:
        print("\nTop results:")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Score: {result.score}")
            print(f"Metadata: {result.payload.get('metadata', {})}")
            print(f"Content preview: {result.payload.get('page_content', '')[:200]}...")

    print("\n2. Testing with filters:")
    
    # Test with Massachusetts team filter
    filter_ma = Filter(
        must=[
            FieldCondition(
                key="metadata.team",
                match=MatchValue(value="Massachusetts - East Region")
            )
        ]
    )

    results_filtered = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
        with_payload=True,
        query_filter=filter_ma
    )
    
    print(f"\nFound {len(results_filtered)} results with Massachusetts filter")
    if results_filtered:
        print("\nTop filtered results:")
        for i, result in enumerate(results_filtered):
            print(f"\n--- Result {i+1} ---")
            print(f"Score: {result.score}")
            print(f"Metadata: {result.payload.get('metadata', {})}")
            print(f"Content preview: {result.payload.get('page_content', '')[:200]}...")

if __name__ == "__main__":
    test_qdrant_queries()
