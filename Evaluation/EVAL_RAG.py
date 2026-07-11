import json
import os
from dotenv import load_dotenv
import sys
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness , answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings

# load retriever and generate_answer function from src/rag.py
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from retriever import Retriever
from rag import generate_answer

# load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path)

with open(os.path.join(os.path.dirname(__file__), "..", "data", "Evalquestion.json")) as f:
    eval_set = json.load(f)


# run each question through pipeline
retriever = Retriever()

rows = []
for item in eval_set:
    q = item["question"]
    results = generate_answer(q, retriever, top_k=6)
    rows.append({
        "question": q,
        "answer": results["answer"],
        "retrieved_contexts": results["context"],
        "ground_truth": item["expected_answer"]
    })
    
dataset = Dataset.from_list(rows)


# 3. RAGAS judge — Groq LLM (free) + your existing Ollama embeddings (free, local)
judge_llm = LangchainLLMWrapper(ChatOpenAI(model="llama-3.3-70b-versatile",
                                           temperature=0.2,
                                           api_key=os.getenv("GROQ_API"),
                                           base_url="https://api.groq.com/openai/v1"))

judge_embedding = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

# run evaluation
results = evaluate(
    dataset = dataset,
    metrics=[faithfulness,answer_relevancy],
    llm = judge_llm,
    embeddings = judge_embedding
)

print("Evaluation Results:",results)
results.to_pandas().to_csv(os.path.join(os.path.dirname(__file__), "..", "data", "eval_results.csv"), index=False)

print("Evaluation completed. Results saved to data/eval_results.csv")