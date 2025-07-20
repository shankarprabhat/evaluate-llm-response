import pandas as pd
import os
import re
import time
import requests # Import requests for Hugging Face API calls
import nest_asyncio
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.settings import Settings
from llama_index.core.workflow import Event, Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.schema import NodeWithScore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage # New imports
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.core.retrievers import VectorIndexRetriever # Added for explicit retriever creation
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from typing import Any, Callable, List, Optional, Sequence, Union
import config

from dotenv import load_dotenv
load_dotenv()  # Load variables from .env into the environment

# Add model mapping
MODELS = {
    "Qwen3": "qwen3:0.6b", # qwen3
    "DeepSeek-R1": "deepseek-r1" # Deepseek
}

# --- New Custom Hugging Face LLM Class ---
class HuggingFaceInferenceLLM(CustomLLM):
    # These fields are defined as Pydantic fields in the class.
    # They will be set by the super().__init__() call.
    base_llm_model_URL: str
    headers: dict
    model_name: str
    llm_selection: str

    def __init__(self, llm_selection: str, **kwargs: Any):
        # 1. Determine the values for the Pydantic fields based on llm_selection
        _base_llm_model_URL: Optional[str] = None
        _headers: Optional[dict] = None
        _model_name: Optional[str] = None

        if llm_selection == 'flan-t5-large':
            token = os.environ.get("HF_TOKEN")
            _TOKEN = "Bearer " + token if token else ""
            _headers = {"Authorization": _TOKEN}
            _base_llm_model_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
            _model_name = "flan-t5-large"
        elif llm_selection == 'hugging-face-qwen':
            _base_llm_model_URL = os.environ.get("HF_QWEN_URL")
            my_hf_model_token = os.environ.get("HF_TOKEN")
            _headers = {
                'Authorization': 'Bearer ' + my_hf_model_token if my_hf_model_token else "",
                'Content-Type': 'application/json'
            }
            _model_name = "hugging-face-qwen"
        elif llm_selection == 'hugging-face-qwen-small':
            _base_llm_model_URL = os.environ.get("HF_QWEN_SMALL_URL")
            my_hf_model_token = os.environ.get("HF_TOKEN")
            _headers = {
                'Authorization': 'Bearer ' + my_hf_model_token if my_hf_model_token else "",
                'Content-Type': 'application/json'
            }
            _model_name = "hugging-face-qwen-small"
        elif llm_selection == 'hugging-face-phi4':
            _base_llm_model_URL = os.environ.get("hugging_face_phi4")
            my_hf_model_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            _headers = {
                'Authorization': 'Bearer ' + my_hf_model_token if my_hf_model_token else "",
                'Content-Type': 'application/json'
            }
            _model_name = "hugging-face-phi4"
        else:
            raise ValueError(f"Unknown LLM selection: {llm_selection}")

        # 2. Perform validation for environment variables *before* Pydantic
        if _base_llm_model_URL is None:
             raise ValueError(f"Environment variable for LLM URL not set for '{llm_selection}'. Expected HF_QWEN_URL or hugging_face_phi4.")
        if _headers is None or not _headers.get('Authorization'):
             raise ValueError(f"Environment variable for LLM token not set for '{llm_selection}'. Expected HF_TOKEN or HUGGINGFACEHUB_API_TOKEN.")
        if _model_name is None:
             raise ValueError(f"Model name not defined for '{llm_selection}'. This indicates an issue in _set_llm_config logic.")

        # 3. Call super().__init__() with the determined values as keyword arguments.
        # This is how Pydantic expects its fields to be initialized.
        # Pass all other kwargs directly to the super constructor as well.
        super().__init__(
            base_llm_model_URL=_base_llm_model_URL,
            headers=_headers,
            model_name=_model_name,
            llm_selection = llm_selection,
            **kwargs
        )
        
        # 4. Set any *non-Pydantic* specific instance variables after super().__init__()
        # In this case, llm_selection is only used internally by your methods,
        # not directly as a Pydantic field that needs validation.
        # self.llm_selection = llm_selection

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=2048,  # Adjust based on your HF model's context window
            num_output=256,       # Adjust based on desired output tokens
            model_name=self.model_name,
        )

    def _get_payload(self, prompt: str, **kwargs: Any) -> dict:
            """Constructs the payload for the Hugging Face Inference API."""
            if self.llm_selection == 'flan-t5-large':
                # Flan-T5 models typically expect a direct string input
                return {"inputs": prompt, "parameters": {"max_new_tokens": self.metadata.num_output, **kwargs}}
            elif self.llm_selection == 'hugging-face-qwen-small':
                # Specific handling for qwen-small if it expects 'inputs' directly
                # This is based on the error message "missing field `inputs`"
                return {
                    "inputs": prompt, # This model might still expect direct 'inputs'
                    "parameters": {
                        "max_new_tokens": self.metadata.num_output,
                        "return_full_text": False,
                        **kwargs
                    }
                }
            elif self.llm_selection in ['hugging-face-qwen', 'hugging-face-phi4']:
                # Larger Qwen and Phi-4 often expect the OpenAI-compatible 'messages' format
                messages_payload = [
                    {"role": "user", "content": prompt}
                ]
                return {
                    "messages": messages_payload,
                    "parameters": {
                        "max_new_tokens": self.metadata.num_output,
                        "return_full_text": False,
                        **kwargs
                    }
                }
            else:
                raise ValueError(f"Unsupported LLM selection for payload construction: {self.llm_selection}")


    def _run_inference_query(self, payload: dict):
        """Runs inference against the Hugging Face endpoint."""
        response = requests.post(self.base_llm_model_URL, headers=self.headers, json=payload)
        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}: {response.text}")
            return None, response.status_code
        try:
            return response.json(), response.status_code
        except requests.exceptions.JSONDecodeError:
            print(f"Failed to decode JSON from response: {response.text}")
            return None, response.status_code


    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completion API for custom LLM."""
        payload = self._get_payload(prompt, **kwargs)
        response_json, status_code = self._run_inference_query(payload)

        if response_json and status_code == 200:
            generated_text = ""
            # The structure of the response might also change for 'messages' API
            # For 'messages' API, it's typically response_json['choices'][0]['message']['content']
            if isinstance(response_json, dict) and 'choices' in response_json and len(response_json['choices']) > 0:
                if 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
                    generated_text = response_json['choices'][0]['message']['content']
            elif isinstance(response_json, list) and len(response_json) > 0 and "generated_text" in response_json[0]:
                # This is for models that return a direct 'generated_text' (like old text generation task)
                generated_text = response_json[0]["generated_text"]
            else:
                generated_text = str(response_json) # Fallback if structure is unexpected

            return CompletionResponse(text=generated_text)
        else:
            return CompletionResponse(text=f"Error: Could not get response from LLM (Status: {status_code})")

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Streaming completion API for custom LLM.
           Note: Hugging Face Inference API generally does not support true streaming out of the box
           for all models in the same way as OpenAI or local Ollama.
           This implementation will get the full response and then yield it as a single chunk.
           For true streaming, you'd need a different Hugging Face Inference Endpoint setup (e.g., TGI with streaming).
        """
        payload = self._get_payload(prompt, **kwargs)
        # Add stream=True if the endpoint supports it and adjust the _run_inference_query
        # to process streaming chunks. For simplicity here, we assume non-streaming then yield.
        # If your endpoint truly streams, you'll need to modify _run_inference_query
        # to yield chunks and modify this method to iterate over them.

        response_json, status_code = self._run_inference_query(payload)

        if response_json and status_code == 200:
            generated_text = ""
            if isinstance(response_json, dict) and 'choices' in response_json and len(response_json['choices']) > 0:
                if 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
                    generated_text = response_json['choices'][0]['message']['content']
            elif isinstance(response_json, list) and len(response_json) > 0 and "generated_text" in response_json[0]:
                generated_text = response_json[0]["generated_text"]
            else:
                generated_text = str(response_json)

            yield CompletionResponse(text=generated_text, delta=generated_text)
        else:
            error_text = f"Error: Could not get streaming response from LLM (Status: {status_code})"
            yield CompletionResponse(text=error_text, delta=error_text)

# --- End of Custom Hugging Face LLM Class ---


class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class RAGWorkflow(Workflow):
    def __init__(self, llm_choice:str, embedding_model="BAAI/bge-large-en-v1.5", persist_dir="./storage"):
        super().__init__()
        print(f"Using LLM model choice: {llm_choice}")
        
        self.llm = None
        
        if llm_choice == 'local_ollama':
            # Use specific model name for Ollama
            ollama_model_name = "qwen3:0.6b" # Default, or pass as an option if needed
            self.llm = Ollama(model=ollama_model_name, request_timeout=300)
            print(f"Ollama model {ollama_model_name} initialized successfully.")
        elif llm_choice in ['flan-t5-large', 'hugging-face-qwen', 'hugging-face-qwen-small','hugging-face-phi4']:
            self.llm = HuggingFaceInferenceLLM(llm_selection=llm_choice)
            print(f"Hugging Face Inference API LLM '{llm_choice}' initialized successfully.")
        else:
            raise ValueError(f"Invalid LLM choice. Supported choices: 'local_ollama', 'flan-t5-large', 'hugging-face-qwen', 'hugging-face-phi4'")
        
        # Get the correct model name based on selection
        # model_name = llm_choice
        # print(f"Using LLM model: {model_name}")
        # if not model_name:
        #     raise ValueError(f"Invalid LLM choice. Please select from: {', '.join(MODELS.keys())}")

        # # Initialize LLM and embedding model
        # self.llm = Ollama(model=model_name,request_timeout=300)
        # print(f"Model {model_name} initialized successfully.")

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

        # Persistence Logic
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"Loading index from {self.persist_dir}...")
            # If storage exists, load the index
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context=storage_context)
            print("Index loaded successfully.")
        else:
            print(f"Creating new index and persisting to {self.persist_dir}...")
            # If no storage, create the index from documents
            documents = SimpleDirectoryReader(dirname).load_data()
            for doc in documents:                
                print(f"Document ID: {doc.id_}, File Name: {doc.metadata.get('file_name')}, File Path: {doc.metadata.get('file_path')}")
            self.index = VectorStoreIndex.from_documents(documents=documents)
            # Persist the index after creation
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print("Index created and persisted successfully.")
        
        return StopEvent(result=self.index)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        """Entry point for RAG retrieval."""
        query = ev.get("query")
        index = ev.get("index") or self.index
        document_filename = ev.get("document_filename") # get document filename if provided
        # print(f"Query: {query}, Index: {index}, Document Filename: {document_filename}")

        fileters =None
        if document_filename:
            # Create a metadata filter to match the document filename
            fileters = MetadataFilters(
                filters=[MetadataFilter(key="file_name",value=document_filename,operator=FilterOperator.IN, case_sensitive=False)
            ],
            condition = FilterCondition.AND
            )
            # print(f"Using metadata filter for document: {document_filename}")
            # print(f"Filters: {fileters}")

        if not query:
            return None

        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = VectorIndexRetriever(index=index, filters=fileters)  # Use explicit retriever
        
        nodes = await retriever.aretrieve(query)
        await ctx.set("query", query)
        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Generate a response using retrieved nodes."""
        summarizer = CompactAndRefine(streaming=True, verbose=True, llm=self.llm)
        query = await ctx.get("query", default=None)
        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)

    async def query(self, query_text: str, document_filename: str = None):
        """Helper method to perform a complete RAG query."""
        if self.index is None:
            raise ValueError("No documents have been ingested. Call ingest_documents first.")

        result = await self.run(query=query_text, index=self.index, document_filename=document_filename)
        return result

    async def ingest_documents(self, directory: str):
        """Helper method to ingest documents."""
        result = await self.run(dirname=directory)
        self.index = result
        return result

# Example usage
async def main(run_config):
    # Initialize the workflow
    workflow = RAGWorkflow(llm_choice=run_config["llm_choice"], embedding_model=run_config["embedding_model"], persist_dir="./storage")

    # Ingest documents
    await workflow.ingest_documents("source")

    df_question = pd.read_excel("Q&A Training Data Use Case 3.xlsx",sheet_name=run_config["sheet_name"])
    question_list = df_question['Question'].tolist()
    ground_truth_list = df_question['Answer'].tolist()

    question_list = question_list           #[:2]  # Limit to first 2 questions for testing
    ground_truth_list = ground_truth_list   #[:2]  # Limit to first 2 answers for testing

    # question_list = question_list[:2]  # Limit to first 2 questions for testing
    # ground_truth_list = ground_truth_list[:2]  # Limit to first 2 answers for testing

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

        try:

            tic = time.time()

            specific_document = None
            specific_document = run_config.get("source_file", None)
            # print(f"Specific document for query: {specific_document}")
            result = await workflow.query(question,document_filename=specific_document)

            # Print the response
            full_response_text = ""
            async for chunk in result.async_response_gen():
                full_response_text += chunk
                # print(chunk, end="", flush=True)

            full_response_text = re.sub(r"<think>.*?</think>", "", full_response_text, flags=re.DOTALL)
            # print(f"\nfull_response_text: {full_response_text}")

            # Extract retrieved contexts (text content only)
            retrieved_contexts = [node.text for node in result.source_nodes]
            retrieved_filenames = [node.metadata.get("file_name", "Unknown") for node in result.source_nodes]
            # print(f"retrieved_contexts: {retrieved_contexts}")
            # print(f"retrieved_filenames: {retrieved_filenames}")
            # Get unique filenames and convert to a list
            unique_retrieved_filenames = list(set(retrieved_filenames))
            print(f"Retrieved from: {', '.join(unique_retrieved_filenames)}") # For console output

            toc = time.time()

            # Add the question and full response to our list
            qa_pairs.append({
                "Question": question,
                "Ground_truth": ground_truth_list[count-2],
                "Response": full_response_text,
                "Contexts": retrieved_contexts,
                "Retrieved_Files": unique_retrieved_filenames,
                "time_taken": round(toc - tic,2)
            })
        except Exception as e:
                print(f"Error processing question '{question}': {e}")
                qa_pairs.append({
                    "Question": question,
                    "Ground_truth": ground_truth_list[count-2] if count-2 < len(ground_truth_list) else "",
                    "Response": "Error processing question",
                    "Contexts": [],
                    "Retrieved_Files": [],
                    "time_taken": 0
                })

        print(f"time taken: {toc-tic} seconds")
    # Create the DataFrame
    qa_df = pd.DataFrame(qa_pairs)

    # Optional: Save the DataFrame to a CSV file    
    csv_filename = os.path.join("output", f"{run_config['sheet_name']}_{run_config['model_name']}_qa_responses.csv")
    # Replace colons in filename to avoid issues on some filesystems
    csv_filename = csv_filename.replace(":", "_")
    qa_df.to_csv(csv_filename, index=False, encoding="utf-8")
    print(f"\nDataFrame saved to {csv_filename}")

if __name__ == "__main__":
    import asyncio
    llm_choice = config.llm_choice
    
    if llm_choice == 'local_ollama':
        model_name = ["qwen3:0.6b","gemma3:1b","deepseek-r1:1.5b"]
        model_name = ["qwen3:0.6b"]
        # model_name = ["gemma3:1b"]
        # model_name = ["deepseek-r1:1.5b"]
    else:
        model_name = ["hugging-face-qwen"]    

    embedding_model = "BAAI/bge-large-en-v1.5"  # Example embedding model
    # embedding_model = "BAAI/bge-small-en-v1.5"  # Example embedding model
    data_sheet = ["21CFR Part 50","21CFR Part 56","21CFR Part 58","NITLT01","AZD9291","ICH GCP E6 R3"]
    source_file = ["21 CFR Part 50.pdf","21 CFR Part 56.pdf","21 CFR Part 58.pdf","NITLT01.pdf","AZD9291.pdf","ich-gcp-r3.pdf"]

    data_sheet = ["21CFR Part 312"]
    source_file = ["21 CFR Part 312.pdf"]

    # data_sheet = ["21CFR Part 50"]
    # source_file = ["21 CFR Part 50.pdf"]

    # data_sheet = ["NITLT01"]
    # source_file = ["NITLT01.pdf"]

    run_config = {
        "llm_choice": llm_choice,
        "model_name": "qwen",  # it will be overwritten
        "sheet_name": "sheet",
        "embedding_model": embedding_model,
        "source_file": source_file
    }

    # Read the source file for question and answer pairs
    for sheet in data_sheet:
        run_config["sheet_name"] = sheet
        for model_llm in model_name:
            run_config["model_name"] = model_llm
            print(f"\nRunning workflow with sheet: {sheet} model: {model_llm}")
            asyncio.run(main(run_config))