import os
import pandas as pd
import json # To handle contexts stored as JSON strings in CSV

from ragas import evaluate
from ragas.llms import LlamaIndexLLMWrapper # To wrap your LLM for Ragas
from ragas.embeddings import LlamaIndexEmbeddingsWrapper # A conceptual wrapper for LlamaIndex embeddings
from datasets import Dataset # Ragas uses Hugging Face Datasets
from ragas.run_config import RunConfig

# Import Ragas evaluation functions
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall, # Can now be used as 'ground_truths' is available!
    # context_relevancy
)

# --- Configuration for Ragas's internal LLM and Embedding Model ---
# (Keep this section as previously provided, ensuring your Ollama LLM and
# FastEmbedEmbedding are correctly instantiated and wrapped for Ragas)
try:
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    from llama_index.core.settings import Settings

    llm_for_ragas = Ollama(model="qwen3:0.6b") # Use your specific Ollama model
    embed_model_for_ragas = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5") 

    ragas_llm = LlamaIndexLLMWrapper(llm_for_ragas)
    ragas_embeddings = LlamaIndexEmbeddingsWrapper(embed_model_for_ragas) 
    
except ImportError:
    print("LlamaIndex or Ragas dependencies not fully installed. Falling back to dummy.")
    class DummyLLM:
        async def agenerate_text(self, prompt: str): return "dummy response"
    class DummyEmbeddings:
        async def aget_query_embedding(self, query: str): return [0.0] * 768

    ragas_llm = LlamaIndexLLMWrapper(DummyLLM())
    ragas_embeddings = LlamaIndexEmbeddingsWrapper(DummyEmbeddings())


# --- Load data from qa_responses.csv ---
try:
    qa_df = pd.read_csv("qa_responses.csv")
    print("Successfully loaded qa_responses.csv")

    # Prepare data for Ragas
    ragas_data_points = []
    for index, row in qa_df.iterrows():
        question = row["Question"]
        response = row["Response"]
        ground_truth = str(row["Ground_truth"]) if "Ground_truth" in row and pd.notna(row["Ground_truth"]) else "" # Ensure string type
        
        # --- Correctly handle 'Contexts' column as a list of strings ---
        contexts = []
        if "Contexts" in row and pd.notna(row["Contexts"]) and row["Contexts"]:
            try:
                # Assuming 'Contexts' column contains a JSON string of a list of strings
                parsed_contexts = json.loads(row["Contexts"]) 
                if isinstance(parsed_contexts, list) and all(isinstance(item, str) for item in parsed_contexts):
                    contexts = parsed_contexts
                elif isinstance(parsed_contexts, str): # If it's a single string, wrap it
                    contexts = [parsed_contexts]
                else:
                    # Fallback for unexpected types after JSON parsing
                    contexts = [str(row["Contexts"])] 
            except (json.JSONDecodeError, TypeError):
                # If it's not a valid JSON string, treat the whole cell as a single context string
                contexts = [str(row["Contexts"])]
        # --- End handling 'Contexts' column ---

        data_point = {
            "question": question,
            "answer": response,  # Your RAG generated response
            "contexts": contexts, # The retrieved document chunks
            "ground_truths": [ground_truth], # Ragas expects a list for ground_truths
            "reference": ground_truth  # Optional, can be used for additional context
        }
        ragas_data_points.append(data_point)

    eval_dataset = Dataset.from_list(ragas_data_points)
    print("Dataset prepared for Ragas evaluation.")

except FileNotFoundError:
    print("Error: qa.csv not found. Please ensure the file exists in the same directory.")
    print("Cannot proceed with Ragas evaluation without the data.")
    exit()
except Exception as e:
    print(f"An error occurred while loading or processing qa.csv: {e}")
    exit()

# --- Ragas Evaluation Configuration ---
ragas_run_config = RunConfig(timeout=2000) 

# --- Perform Ragas Evaluation ---
print("\n--- Starting Ragas Evaluation ---")
score = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall # Now enabled!
        # context_relevancy,
    ],
    llm=ragas_llm, 
    embeddings=ragas_embeddings,
    run_config=ragas_run_config # --- ADDED: Pass the run_config ---
)

print("\n--- Ragas Evaluation Results ---")
print(score)
print("\nDetailed scores:")
print(score.to_pandas())
results_df = score.to_pandas()
# Save the results to a CSV file  
results_csv_filename = "ragas_evaluation_results.csv"
results_df.to_csv(results_csv_filename, index=False, encoding="utf-8")