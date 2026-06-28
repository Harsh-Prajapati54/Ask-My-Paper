import os
from dotenv import load_dotenv
from loader import *
from chunking import *
from Embedding import *
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams,PointStruct

load_dotenv()
Qdrant_Api = os.getenv("QDRANT_API")
Qdrant_Url = os.getenv("QDRANT_URL")
# Connect to Qdrant Cloud
client = QdrantClient(
    url= Qdrant_Url,
    api_key=Qdrant_Api
)
#calling embedding function

client.delete_collection("DOCS")  # run this once
COLLECTION_NAME = "DOCS" # name of collection
# create Collection 
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance = Distance.COSINE   
    )
)

def upsert_embeddings(client, collection_name: str, embeddings: list, chunks: list, batch_size: int = 100):

    points = [
    PointStruct(
        id=i,
        vector=embeddings[i],
        payload={"text": chunks[i]}  # store anything here
    )
    for i in range(len(embeddings))
    ]
    
    for i in range(0,len(points),batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
)

embeddings, chunks = embed_doc(file_path)
print(type(embeddings[0]))      # should be list, not str
print(embeddings[0][:3])        # should be [0.123, 0.456, 0.789], not text
upsert_embeddings(client, "DOCS", embeddings, chunks)

query_vector = embed_query("what is attention mechanism?")  # your embedding function

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=5,
    with_payload=True
)

for r in results:
    print(r.score, r.payload["text"])