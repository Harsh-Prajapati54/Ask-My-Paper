#imports 
import pandas as pd
from vectordb import *
from loader import *
from chunking import *
from rank_bm25 import BM25Okapi
from flashrank import Ranker , RerankRequest

class Retriever:
    def __init__(self, chunks: list[str], rrf_k:int = 60):
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([c.lower().split() for c in chunks])
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        
    def dense_rank(self, query: str, top_k:int = 25) -> list[int]:
        dense_results = dense_search(query, top_k=top_k)
        return dense_results["id"].tolist()
    
    def bm25_rank(self, query: str, top_k:int) -> list[int]:
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        ranked_ids = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        
        return ranked_ids[:top_k]
    
    def rrf_rank(self, dense_ids:list[int],bm25_ids:list[int]) -> list[int]:
        rrf_scores = {}
        for rank, doc_id in enumerate(dense_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 1 + self.rrf_k)
        
        for rank, doc_id in enumerate(bm25_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 1 + self.rrf_k)
        
        ranked_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        return ranked_ids
    
    def retrieve(self,query:str,top_k:int = 6,fusion_pool: int = 25) -> pd.DataFrame:
        dense_ids = self.dense_rank(query, fusion_pool)
        bm25_ids = self.bm25_rank(query, fusion_pool)
        final_ids = self.rrf_rank(dense_ids, bm25_ids)[:fusion_pool]
        
        passages = [{"id":id,"text":self.chunks[id]} for id in final_ids]
        reranked = self.reranker.rerank(RerankRequest(query=query, passages=passages))
        
        return pd.DataFrame(reranked[:top_k])
    
    
if __name__ == "__main__":
    chunks = chunk_document(file_path)
    retriever = Retriever(chunks)
    print(retriever.retrieve("what is prompt engineering?"))