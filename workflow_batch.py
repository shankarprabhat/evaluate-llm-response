import pandas as pd
import os
import re
import time
import nest_asyncio
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.settings import Settings
from llama_index.core.workflow import Event, Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.schema import NodeWithScore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage # New imports
from llama_index.core.response_synthesizers import CompactAndRefine

# Add model mapping
MODELS = {
    "Qwen3": "qwen3:0.6b", # qwen3
    "DeepSeek-R1": "deepseek-r1" # Deepseek
}

class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class RAGWorkflow(Workflow):
    def __init__(self, llm_choice="Qwen3", embedding_model="BAAI/bge-large-en-v1.5", persist_dir="./storage"):
        super().__init__()
        # Get the correct model name based on selection
        # model_name = MODELS.get(llm_choice)
        model_name = llm_choice
        print(f"Using LLM model: {model_name}")
        if not model_name:
            raise ValueError(f"Invalid LLM choice. Please select from: {', '.join(MODELS.keys())}")
            
        # Initialize LLM and embedding model
        self.llm = Ollama(model=model_name,request_timeout=300)
        print(f"Model {model_name} initialized successfully.")

        # Configure a custom cache directory for fastembed (recommended)
        custom_cache_dir = os.path.join(os.getcwd(), "fastembed_models_cache")
        os.makedirs(custom_cache_dir, exist_ok=True)
        self.embed_model = FastEmbedEmbedding(model_name=embedding_model, cache_dir=custom_cache_dir)

        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self.index = None
        self.persist_dir = persist_dir # Store the persistence directory

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest documents from a directory."""
        dirname = ev.get("dirname")
        if not dirname:
            return None
        
        # --- MODIFIED: Persistence Logic ---
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"Loading index from {self.persist_dir}...")
            # If storage exists, load the index
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
            print("Index loaded successfully.")
        else:
            print(f"Creating new index and persisting to {self.persist_dir}...")
            # If no storage, create the index from documents
            documents = SimpleDirectoryReader(dirname).load_data()
            self.index = VectorStoreIndex.from_documents(documents=documents)
            # Persist the index after creation
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print("Index created and persisted successfully.")

        # documents = SimpleDirectoryReader(dirname).load_data()
        # self.index = VectorStoreIndex.from_documents(documents=documents)
        return StopEvent(result=self.index)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        """Entry point for RAG retrieval."""
        query = ev.get("query")
        index = ev.get("index") or self.index

        if not query:
            return None

        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = index.as_retriever(similarity_top_k=2)
        nodes = await retriever.aretrieve(query)
        await ctx.set("query", query)
        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Generate a response using retrieved nodes."""
        summarizer = CompactAndRefine(streaming=True, verbose=True)
        query = await ctx.get("query", default=None)
        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)

    async def query(self, query_text: str):
        """Helper method to perform a complete RAG query."""
        if self.index is None:
            raise ValueError("No documents have been ingested. Call ingest_documents first.")
        
        result = await self.run(query=query_text, index=self.index)
        return result

    async def ingest_documents(self, directory: str):
        """Helper method to ingest documents."""
        result = await self.run(dirname=directory)
        self.index = result
        return result

# Example usage
async def main(run_config):
    # Initialize the workflow
    workflow = RAGWorkflow(llm_choice=run_config["model_name"], embedding_model=run_config["embedding_model"], persist_dir="./storage")
    
    # Ingest documents
    await workflow.ingest_documents("source")

    df_question = pd.read_excel("Q&A Training Data Use Case 3.xlsx",sheet_name=run_config["sheet_name"])
    question_list = df_question['Question'].tolist()
    ground_truth_list = df_question['Answer'].tolist()

    question_list = question_list     #[:2]  # Limit to first 2 questions for testing
    ground_truth_list = ground_truth_list  #[:2]  # Limit to first 2 answers for testing

    # List to store dictionaries of questions and responses
    qa_pairs = [] 
    
    # question_list = [
    #     "What is the purpose of DeepSeekR1?",
    #     "How was DeepSeekR1 trained?",
    #     "What are the main features of DeepSeekR1?",
    #     "What datasets were used to train DeepSeekR1?",
    #     "How does DeepSeekR1 compare to other models?"
    # ]
    count = 1
    for question in question_list:
        print(f"\n{count}--- Query: {question} ---")
        count += 1
        tic = time.time()
        try:
            result = await workflow.query(question)
            
            # Print the response
            full_response_text = ""        
            async for chunk in result.async_response_gen():            
                full_response_text += chunk
                # print(chunk, end="", flush=True)

            full_response_text = re.sub(r"<think>.*?</think>", "", full_response_text, flags=re.DOTALL)
            # print(f"\nfull_response_text: {full_response_text}")
            
            # Extract retrieved contexts (text content only)
            retrieved_contexts = [node.text for node in result.source_nodes]

            # Add the question and full response to our list
            qa_pairs.append({
                "Question": question,
                "Ground_truth": ground_truth_list[count-2],
                "Response": full_response_text,
                "Contexts": retrieved_contexts            
            })
        except Exception as e:
                print(f"Error processing question '{question}': {e}")
                qa_pairs.append({
                    "Question": question,
                    "Ground_truth": ground_truth_list[count-2] if count-2 < len(ground_truth_list) else "",
                    "Response": "Error processing question",
                    "Contexts": []
                })  

        toc = time.time()
        print(f"time taken: {toc-tic} seconds")    
    # Create the DataFrame
    qa_df = pd.DataFrame(qa_pairs)
    
    # Optional: Save the DataFrame to a CSV file
    csv_filename = "qa_responses.csv"
    csv_filename = os.path.join("output", f"{run_config['sheet_name']}_{run_config['model_name']}_qa_responses.csv")    
    # Replace colons in filename to avoid issues on some filesystems
    csv_filename = csv_filename.replace(":", "_")  
    qa_df.to_csv(csv_filename, index=False, encoding="utf-8")
    print(f"\nDataFrame saved to {csv_filename}")

if __name__ == "__main__":
    import asyncio

    model_name = ["qwen3:0.6b","gemma3:1b","deepseek-r1:1.5b"]
    # model_name = ["gemma3:1b"]
    # model_name = ["deepseek-r1:1.5b"]
    embedding_model = "BAAI/bge-large-en-v1.5"  # Example embedding model
    data_sheet = ["21CFR Part 50","21CFR Part 56","21CFR Part 58","NITLT01","AZD9291","ICH GCP E6 R3"]
    source_file = ["21CFR Part 50.pdf","21CFR Part 56.pdf","21CFR Part 58.pdf","NITL01.pdf","AZD9291.pdf"]
    
    run_config = {
        "model_name": "qwen",
        "sheet_name": "sheet",
        "embedding_model": embedding_model
    }
    
    # Read the source file for question and answer pairs
    for sheet in data_sheet:        
        run_config["sheet_name"] = sheet        
        for model_llm in model_name:
            run_config["model_name"] = model_llm
            print(f"\nRunning workflow with sheet: {sheet} model: {model_llm}")
            asyncio.run(main(run_config))