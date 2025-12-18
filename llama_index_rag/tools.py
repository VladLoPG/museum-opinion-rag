import os
from datetime import date

from dotenv import load_dotenv
import pandas as pd
import uuid

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.llms.llama_cpp import LlamaCPP


def load_csv(path: str) -> pd.DataFrame:
    """
    Loads reviews from .csv
    """
    data = pd.read_csv(path, index_col=0)
    st.success("Data loaded successfully")
    return data


def create_json(data: pd.DataFrame) -> dict[list]:
    """
    Forms json based on the table

    :param data: Takes DataFrame from the previous step
    :type data: pd.DataFrame
    :return: dictionary with 'documents' key and a value of a list of json objects
    :rtype: dict
    """

    documents = []
    for _, row in data.iterrows():
        doc = {}
        doc["id"] = str(uuid.uuid4())
        doc["meta"] = {
            "object": row["object"],
            "year": row["publication_year"],
            "platform": row["platform"],
        }
        doc["text"] = row["text"]
        doc["created_date"] = date.today().strftime("%d.%m.%Y")
        documents.append(doc)

    st.success("Main json created")
    return {"documents": documents}


def chunk_dataset(json_data: dict[list]) -> dict[list]:
    """
    Creates json based on recursive chunking of the main json with full reviews

    :param json_data: Has info about the original review plus chunk metadata
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)

    chunks = []

    for doc in json_data["documents"]:
        all_chunks = splitter.split_text(doc["text"])

        for i, chunk_text in enumerate(all_chunks):
            chunk = {}
            chunk["id"] = f"{doc['id']}_chunk_{str(i)}"
            chunk["original_doc_id"] = doc["id"]
            chunk["meta"] = doc["meta"]
            chunk["text"] = chunk_text
            chunk["chunk_info"] = {
                "position": i,
                "total_chunks": len(all_chunks),
            }
            chunks.append(chunk)

    st.success("Chunks created")
    return {"chunks": chunks}


def get_documents(
    chunked_data,
    excluded_llm=["chunk_id", "chunk_position"],
    excluded_embed=["chunk_id", "chunk_position", "total_chunks"],
):
    """
    Forms llamaindex Documents based on chunked json

    :param chunked_data: json from previous step
    :param excluded_llm: what data to hide from llm
    :param excluded_embed: what data exclude from embeddings
    """
    documents = [
        Document(
            text=item["text"],
            metadata={
                "object": item["meta"]["object"],
                "platform": item["meta"]["platform"],
                "year": item["meta"]["year"],
                "chunk_id": item["id"],  
                "original_doc": item["original_doc_id"],
                "chunk_position": item["chunk_info"][
                    "position"
                ],  
                "total_chunks": item["chunk_info"][
                    "total_chunks"
                ],  
            },
            
            excluded_llm_metadata_keys=excluded_llm,
            excluded_embed_metadata_keys=excluded_embed,
        )
        for item in chunked_data["chunks"]
    ]
    st.success("LlamaIndex Documents created")
    return documents


def set_vector_store(
    documents,
    collection_name,
    embed_model_name="paraphrase-multilingual-MiniLM-L12-v2",
):
    """
    Creates new ChromaDB collection

    :param documents: list of llamaindex docs
    :param collection_name: new collection name
    :param embed_model_name: embedding model name from hf
    """

    st.text(f"Creating new collection: {collection_name}")
    embed_model = HuggingFaceEmbedding(embed_model_name)
    chroma_client = chromadb.PersistentClient(path="./chroma")
    chroma_collection = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True,
    )
    st.success("Vector storage loaded")

    return index


def get_vector_store(
    collection_name, embed_model_name="paraphrase-multilingual-MiniLM-L12-v2"
):
    """
    Loads existing collection

    :param collection_name: collection name
    :param embed_model_name: hf model name (needed for compatibility)
    """
    st.text(f"Loading existing collection: {collection_name}")
    chroma_client = chromadb.PersistentClient(path="./chroma")
    embed_model = HuggingFaceEmbedding(embed_model_name)
    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    st.success("Vector storage loaded")

    return index


@st.cache_resource
def set_llm(_index, inference="Groq", model="llama-3.1-8b-instant"):
    """
    Configures LLM query engine

    :param index: ChromaDB collection index
    :param inference: Inference engine (Groq or llama)
    :param model: model name (Groq) or model path (llama)
    """

    system_prompt = """
    You are a professional museum review analyst. 
    Answer the user's question based ONLY on the provided context of visitor reviews. 
    If the context doesn't contain the answer, say that you cannot answer this question based on the provided reviews. 
    Only use the Russian language in your answers. Always use markdown
    """

    if inference == "llamacpp":
        llm = LlamaCPP(
            model_path=model,
            temperature=0.3,
            context_window=8192,
            max_new_tokens=512,
            model_kwargs={"n_gpu_layers": -1, "n_ctx": 8192},
            system_prompt=system_prompt,
            verbose=False,
        )

    else:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        llm = Groq(model=model, api_key=api_key, system_prompt=system_prompt)

    query_engine = _index.as_query_engine(llm=llm, similarity_top_k=15)

    st.success("LLM query engine loaded")
    return query_engine


def get_response(query_engine, query):
    """
    Receive answer based on query, also get chunks in descending order of confidence scores

    :param query_engine: Поисковый движок с ллм
    :param query: Пользовательский запрос
    """

    response = query_engine.query(query)

    result = {
        "text": response.response,
        "nodes": response.source_nodes,
    }

    st.text("Model answer")
    st.markdown(result["text"])
    st.text(f"\nRelevant chunks:")
    st.markdown("---")

    for i, node in enumerate(result["nodes"], start=1):
        st.text(
            f"Chunk №{i}, score = {node.score:.3f}, object = {node.metadata.get('object')}, year = {node.metadata.get('year')}"
        )
        st.text(node.node.get_content())
        st.markdown("---")
