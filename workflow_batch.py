import pandas as pd
import os
import re
import time
import requests
import nest_asyncio
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.settings import Settings
from llama_index.core.workflow import Event, Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.schema import NodeWithScore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from typing import Any, Callable, List, Optional, Sequence, Union
from llama_index.core.prompts import PromptTemplate
import config

from dotenv import load_dotenv
load_dotenv()

# Add model mapping (not directly used by HuggingFaceInferenceLLM's internal logic, but kept for context)
MODELS = {
    "Qwen3": "qwen3:0.6b",
    "DeepSeek-R1": "deepseek-r1"
}

"""
## Custom Hugging Face LLM Class

This class is enhanced to accept and manage LLM generation parameters directly.

"""
class HuggingFaceInferenceLLM(CustomLLM):
    base_llm_model_URL: str
    headers: dict
    model_name: str
    llm_selection: str

    # **NEW:** Define generation parameters as Optional Pydantic fields
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    # Add any other parameters you might want to control, e.g.,
    # max_new_tokens: Optional[int] = None # already handled by metadata.num_output, but can be explicit

    def __init__(
        self,
        llm_selection: str,
        # **NEW:** Accept generation parameters in the constructor
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs: Any
    ):
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
            _base_llm_model_URL = os.environ.get("HF_PHI4_URL")
            my_hf_model_token = os.environ.get("HF_TOKEN")
            _headers = {
                'Authorization': 'Bearer ' + my_hf_model_token if my_hf_model_token else "",
                'Content-Type': 'application/json'
            }
            _model_name = "hugging-face-phi4"
        else:
            raise ValueError(f"Unknown LLM selection: {llm_selection}")

        if _base_llm_model_URL is None:
             raise ValueError(f"Environment variable for LLM URL not set for '{llm_selection}'. Expected HF_QWEN_URL or hugging_face_phi4.")
        if _headers is None or not _headers.get('Authorization'):
             raise ValueError(f"Environment variable for LLM token not set for '{llm_selection}'. Expected HF_TOKEN or HUGGINGFACEHUB_API_TOKEN.")
        if _model_name is None:
             raise ValueError(f"Model name not defined for '{llm_selection}'. This indicates an issue in _set_llm_config logic.")

        # **MODIFIED:** Pass generation parameters to super().__init__()
        super().__init__(
            base_llm_model_URL=_base_llm_model_URL,
            headers=_headers,
            model_name=_model_name,
            llm_selection=llm_selection,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            **kwargs
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=2048,
            num_output=256,
            model_name=self.model_name,
        )

    def _get_payload(self, prompt: str, **kwargs: Any) -> dict:
        """Constructs the payload for the Hugging Face Inference API."""

        # **MODIFIED:** Prioritize instance attributes, then kwargs, then defaults
        generation_params = {
            "max_new_tokens": self.metadata.num_output,
            "return_full_text": False,
            "temperature": self.temperature,        # Get from instance attribute
            "top_k": self.top_k,                    # Get from instance attribute
            "top_p": self.top_p,                    # Get from instance attribute
            "repetition_penalty": self.repetition_penalty, # Get from instance attribute
            "do_sample": self.do_sample,            # Get from instance attribute
        }

        # Override with any parameters passed explicitly to complete/stream_complete (kwargs)
        generation_params.update(kwargs)

        # Remove None values so they don't get sent to the API
        generation_params = {k: v for k, v in generation_params.items() if v is not None}

        # Set good defaults only if not overridden by instance attributes or kwargs
        if generation_params.get("do_sample", True): # default to sampling if not explicitly set to False
            generation_params.setdefault("temperature", 0.7)
            generation_params.setdefault("top_p", 0.9)
            generation_params.setdefault("top_k", 50)
            generation_params.setdefault("repetition_penalty", 1.2)
            generation_params.setdefault("no_repeat_ngram_size", 3) # Specific to some HF models

        print(f"HuggingFaceInferenceLLM sending parameters: {generation_params}")

        if self.llm_selection == 'flan-t5-large':
            return {"inputs": prompt, "parameters": generation_params} # Pass generation_params here
        elif self.llm_selection == 'hugging-face-qwen-small':
            return {
                "inputs": prompt,
                "parameters": generation_params
            }
        elif self.llm_selection in ['hugging-face-qwen', 'hugging-face-phi4']:
            messages_payload = [
                {"role": "user", "content": prompt}
            ]
            return {
                "messages": messages_payload,
                "parameters": generation_params
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
            if isinstance(response_json, dict) and 'choices' in response_json and len(response_json['choices']) > 0:
                if 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
                    generated_text = response_json['choices'][0]['message']['content']
            elif isinstance(response_json, list) and len(response_json) > 0 and "generated_text" in response_json[0]:
                generated_text = response_json[0]["generated_text"]
            else:
                generated_text = str(response_json)

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
            
class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class RAGWorkflow(Workflow):
    def __init__(
        self,
        llm_choice: str,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        persist_dir: str = "./storage",
        # **NEW:** Accept generation parameters here with default values
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs: Any # For any other general LLM initialization kwargs
    ):
        super().__init__()
        print(f"Using LLM model choice: {llm_choice}")

        self.llm = None

        if llm_choice == 'local_ollama':
            ollama_model_name = "qwen3:0.6b"
            self.llm = Ollama(model=ollama_model_name, request_timeout=300)
            print(f"Ollama model {ollama_model_name} initialized successfully.")
        elif llm_choice in ['flan-t5-large', 'hugging-face-qwen', 'hugging-face-qwen-small', 'hugging-face-phi4']:
            # **MODIFIED:** Pass the generation parameters to the HuggingFaceInferenceLLM constructor
            self.llm = HuggingFaceInferenceLLM(
                llm_selection=llm_choice,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                **kwargs # Pass any extra kwargs that HuggingFaceInferenceLLM might accept
            )
            print(f"Hugging Face Inference API LLM '{llm_choice}' initialized successfully with custom parameters.")
        else:
            raise ValueError(f"Invalid LLM choice. Supported choices: 'local_ollama', 'flan-t5-large', 'hugging-face-qwen', 'hugging-face-phi4'")

        custom_cache_dir = os.path.join(os.getcwd(), "fastembed_models_cache")
        os.makedirs(custom_cache_dir, exist_ok=True)
        self.embed_model = FastEmbedEmbedding(model_name=embedding_model, cache_dir=custom_cache_dir)

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.index = None
        self.persist_dir = persist_dir

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest documents from a directory."""
        dirname = ev.get("dirname")
        if not dirname:
            return None

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"Loading index from {self.persist_dir}...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context=storage_context)
            print("Index loaded successfully.")
        else:
            print(f"Creating new index and persisting to {self.persist_dir}...")
            documents = SimpleDirectoryReader(dirname).load_data()
            for doc in documents:
                print(f"Document ID: {doc.id_}, File Name: {doc.metadata.get('file_name')}, File Path: {doc.metadata.get('file_path')}")
            self.index = VectorStoreIndex.from_documents(documents=documents)
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print("Index created and persisted successfully.")

        return StopEvent(result=self.index)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        """Entry point for RAG retrieval."""
        query = ev.get("query")
        index = ev.get("index") or self.index
        document_filename = ev.get("document_filename")

        filters = None
        if document_filename:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="file_name", value=document_filename, operator=FilterOperator.IN, case_sensitive=False)],
                condition=FilterCondition.AND
            )

        if not query:
            return None

        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = VectorIndexRetriever(index=index, filters=filters)
        nodes = await retriever.aretrieve(query)
        await ctx.set("query", query)
        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        
        """Generate a response using retrieved nodes."""
        # --- THIS IS WHERE WE INJECT CUSTOM PROMPTS ---
        # Define custom QA and Refine templates
        custom_qa_prompt_tmpl = PromptTemplate(
            """
            Context information is below.
            ---------------------
            {context_str}
            ---------------------
            Given the context information and not prior knowledge,
            you MUST think step-by-step how to answer the question, then provide your final answer.
            Crucially, your final answer should ONLY contain the answer itself, without any of your thinking process or leading phrases like 'Final Answer:'.
            Ensure the answer is comprehensive, accurate, and directly addresses the question.

            Question: {query_str}
            """
        )

        custom_refine_prompt_tmpl = PromptTemplate(
            """
            The original question is as follows: {query_str}
            We have provided an existing answer: {existing_answer}
            We have some more context including the original question and the new retrieved document to refine the original answer:
            ------------
            {context_msg}
            ------------
            Given the new context, refine the original answer to better answer the question.
            If the context isn't useful, return the original answer.
            You MUST think step-by-step how to refine the answer, then provide your final answer.
            Crucially, your refined answer should ONLY contain the answer itself, without any of your thinking process or leading phrases like 'Refined Answer:'.
            Ensure the answer is comprehensive, accurate, and directly addresses the question.

            Refined Answer:
            """
        )


        summarizer = CompactAndRefine(
            streaming=True,
            verbose=True,
            llm=self.llm,
            text_qa_template=custom_qa_prompt_tmpl, # Pass custom QA template
            refine_template=custom_refine_prompt_tmpl # Pass custom Refine template
        )
        query = await ctx.get("query", default=None)
        response = await summarizer.asynthesize(query, nodes=ev.nodes)        
        
        # """Generate a response using retrieved nodes."""
        # summarizer = CompactAndRefine(streaming=True, verbose=True, llm=self.llm)
        # query = await ctx.get("query", default=None)
        # response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)

    async def query(self, query_text: str, document_filename: str = None):
        """Helper method to perform a complete RAG query.
           **Note**: LLM generation parameters are now configured during RAGWorkflow initialization.
        """
        if self.index is None:
            raise ValueError("No documents have been ingested. Call ingest_documents first.")

        # The 'run' method of the workflow does not need LLM specific parameters.
        # It just needs the context for the steps (query, index, filename).
        result = await self.run(query=query_text, index=self.index, document_filename=document_filename)
        return result

    async def ingest_documents(self, directory: str):
        """Helper method to ingest documents."""
        result = await self.run(dirname=directory)
        self.index = result
        return result

async def main(run_config):
    # Initialize the workflow and pass the LLM generation parameters here
    workflow = RAGWorkflow(
        llm_choice=run_config["llm_choice"],
        embedding_model=run_config["embedding_model"],
        persist_dir="./storage",
        # **MODIFIED:** Pass generation parameters to RAGWorkflow's constructor
        temperature=run_config.get("llm_temperature", 0.7),
        top_k=run_config.get("llm_top_k", 50),
        top_p=run_config.get("llm_top_p", 0.9),
        repetition_penalty=run_config.get("llm_repetition_penalty", 1.2),
        do_sample=run_config.get("llm_do_sample", True),
        # You can add other LLM specific parameters here if needed
    )

    # Ingest documents
    await workflow.ingest_documents("source")

    df_question = pd.read_excel("QA_Training_Data_Use_Case_3.xlsx", sheet_name=run_config["sheet_name"])
    question_list = df_question['Question'].tolist()
    ground_truth_list = df_question['Answer'].tolist()

    question_list = question_list   #[:10]
    ground_truth_list = ground_truth_list    #[:10]

    qa_pairs = []
    count = 1
    for question in question_list:
        print(f"\n{count}--- Query: {question} ---")
        count += 1

        try:
            tic = time.time()

            specific_document = run_config.get("source_file", None)

            # **MODIFIED:** DO NOT pass generation parameters to workflow.query() anymore
            result = await workflow.query(
                question,
                document_filename=specific_document,
            )

            full_response_text = ""
            async for chunk in result.async_response_gen():
                full_response_text += chunk

            full_response_text = re.sub(r"<think>.*?</think>", "", full_response_text, flags=re.DOTALL)

            retrieved_contexts = [node.text for node in result.source_nodes]
            retrieved_filenames = [node.metadata.get("file_name", "Unknown") for node in result.source_nodes]

            unique_retrieved_filenames = list(set(retrieved_filenames))
            print(f"Retrieved from: {', '.join(unique_retrieved_filenames)}")

            toc = time.time()

            qa_pairs.append({
                "Question": question,
                "Ground_truth": ground_truth_list[count-2] if count-2 < len(ground_truth_list) else "",
                "Response": full_response_text,
                "Contexts": retrieved_contexts,
                "Retrieved_Files": unique_retrieved_filenames,
                "time_taken": round(toc - tic, 2)
            })
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            time_taken_on_error = round(time.time() - tic, 2) if tic != 0.0 else 0.0
            qa_pairs.append({
                "Question": question,
                "Ground_truth": ground_truth_list[count-2] if count-2 < len(ground_truth_list) else "",
                "Response": "Error processing question",
                "Contexts": [],
                "Retrieved_Files": [],
                "time_taken": time_taken_on_error
            })

        print(f"time taken: {toc-tic} seconds")

    qa_df = pd.DataFrame(qa_pairs)

    csv_filename = os.path.join("output", f"{run_config['sheet_name']}_{run_config['model_name']}_qa_responses.csv")
    csv_filename = csv_filename.replace(":", "_")
    qa_df.to_csv(csv_filename, index=False, encoding="utf-8")
    print(f"\nDataFrame saved to {csv_filename}")

if __name__ == "__main__":
    import asyncio
    llm_choice = config.llm_choice

    if llm_choice == 'local_ollama':
        model_name = ["qwen3:0.6b"] # Keep only one for testing, or iterate as before
    else:
        model_name = ["hugging-face-qwen"]

    embedding_model = "BAAI/bge-large-en-v1.5"
    data_sheet = ["21CFR Part 312"]
    source_file = ["21 CFR Part 312.pdf"]

    for sheet, file in zip(data_sheet, source_file): # Use zip to iterate through sheets and files together
        for model_llm in model_name:
            # **MODIFIED:** Create run_config inside the loop and add LLM parameters
            run_config = {
                "llm_choice": llm_choice,
                "model_name": model_llm, # This will now be the actual model name being run
                "sheet_name": sheet,
                "embedding_model": embedding_model,
                "source_file": file, # Pass the corresponding file
                # **NEW:** Add desired LLM generation parameters to run_config
                "llm_temperature": 0.8,
                "llm_top_k": 40,
                "llm_top_p": 0.9,
                "llm_repetition_penalty": 1.2,
                "llm_do_sample": False,
            }
            print(f"\nRunning workflow with sheet: {sheet}, model: {model_llm}, file: {file}")
            asyncio.run(main(run_config))