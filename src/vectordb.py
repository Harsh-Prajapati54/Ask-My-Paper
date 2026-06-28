from loader import *
from chunking import *
from Embedding import *
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams,PointStruct,Document
import ollama

# Connect to Qdrant Cloud
client = QdrantClient(
    url= "https://1b80c87a-05f2-49a5-81bd-e145c0d5287c.us-east-2-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZTA1MmVlYTktYzA3Yi00MTMzLWI2MmQtNjg5Zjk3ZDQ2NjhhIn0.r46jfIWu3lXFvYRhehXpPm7gSW42QwDRWb3Lw3F55f8",
    cloud_inference= True
)
#calling embedding function
embedding = embed_doc(file_path)
Embedding_dim = len(embedding[0])

COLLECTION_NAME = "book" # name of collection
# create Collection 
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=Embedding_dim,
        distance = Distance.COSINE   
    )
)


input_text = chunk_document(file_path)

points = [
    PointStruct(
        id=i,
        vector=embedding[i],
        payload={"text": input_text[i]}  # store anything here
    )
    for i in range(len(embedding))
]

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

query_vector = embed_doc("what is attention mechanism?")  # your embedding function

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=5,
    with_payload=True
)

for r in results:
    print(r.score, r.payload["text"])