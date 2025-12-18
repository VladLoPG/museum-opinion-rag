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
    Загружает таблицу с отзывами из .csv
    """
    data = pd.read_csv(path, index_col=0)
    st.success("Данные загружены успешно")
    return data


def create_json(data: pd.DataFrame) -> dict[list]:
    """
    Формирует json на основе данных из таблицы

    :param data: Принимает датафрейм с предыдущей функции
    :type data: pd.DataFrame
    :return: словарь с ключом 'documents' и значением в виде списка
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

    st.success("Основной жосон создан")
    return {"documents": documents}


def chunk_dataset(json_data: dict[list]) -> dict[list]:
    """
    Создает json на основе целых отзывов, производя рекурсивное чанкирование

    :param json_data: Содержит информацию оригинального отзыва + метаданные чанка
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

    st.success("Чанки созданы")
    return {"chunks": chunks}


def get_documents(
    chunked_data,
    excluded_llm=["chunk_id", "chunk_position"],
    excluded_embed=["chunk_id", "chunk_position", "total_chunks"],
):
    """
    Формирует Document для ллама индекс на основе json чанков

    :param chunked_data: json с предыдущего шага
    :param excluded_llm: какие данные скрыть от ллм
    :param excluded_embed: какие данные не учитывать в эмбеддингах
    """
    documents = [
        Document(
            text=item["text"],
            metadata={
                "object": item["meta"]["object"],
                "platform": item["meta"]["platform"],
                "year": item["meta"]["year"],
                "chunk_id": item["id"],  # Только для отладки
                "original_doc": item["original_doc_id"],  # Для трассировки источников
                "chunk_position": item["chunk_info"][
                    "position"
                ],  # Техническая информация
                "total_chunks": item["chunk_info"][
                    "total_chunks"
                ],  # Техническая информация
            },
            # Что скрыть от LLM при генерации ответов (конфиденциальные, избыточные данные)
            excluded_llm_metadata_keys=excluded_llm,
            # Что исключить из векторных эмбеддингов (не влияет на поиск векторного сходства)
            excluded_embed_metadata_keys=excluded_embed,
        )
        for item in chunked_data["chunks"]
    ]
    st.success("Документы лламаиндекс созданы")
    return documents


def set_vector_store(
    documents,
    collection_name,
    embed_model_name="paraphrase-multilingual-MiniLM-L12-v2",
):
    """
    Создает новую коллекцию в БД

    :param documents: список документов ллама индекс
    :param collection_name: название новой коллекции
    :param embed_model_name: название эмбеддинг модели из huggingface
    """

    st.text(f"Делаю новую коллекцию: {collection_name}")
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
    st.text("Векторное хранилище загружено")

    return index


def get_vector_store(
    collection_name, embed_model_name="paraphrase-multilingual-MiniLM-L12-v2"
):
    """
    Загружает существующую коллекцию

    :param collection_name: название коллекции
    :param embed_model_name: название модели huggingface, которая делала эмбеддинги (нужно для совместимости поиска)
    """
    st.text(f"Загружаю существующую коллекцию: {collection_name}")
    chroma_client = chromadb.PersistentClient(path="./chroma")
    embed_model = HuggingFaceEmbedding(embed_model_name, device="cpu")
    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    st.success("Векторное хранилище загружено")

    return index


@st.cache_resource
def set_llm(_index, inference="Groq", model="llama-3.1-8b-instant"):
    """
    Настраивает поисковый движок с ллм, по умолчанию использует базовую модель из апи Groq (нужен файл .env в рабочей директории)
    Также есть опция локального инференса через llamacpp, тогда нужен путь к модели формата gguf

    :param index: Индекс коллекции хромадб
    :param inference: Движок для инференса - Groq или llamacpp
    :param model: Название модели (Groq) или путь к модели (llamacpp)
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

    st.success("Поисковый движок с ллм загружен")
    return query_engine


def get_response(query_engine, query):
    """
    Получаем ответ на запрос, выводится как ответ модели, так и релевантные чанки в порядке confidence score

    :param query_engine: Поисковый движок с ллм
    :param query: Пользовательский запрос
    """

    response = query_engine.query(query)

    result = {
        "text": response.response,
        "nodes": response.source_nodes,
    }

    st.text("Ответ модели")
    st.markdown(result["text"])
    st.text(f"\nРелевантные чанки:")
    st.markdown("---")

    for i, node in enumerate(result["nodes"], start=1):
        st.text(
            f"Чанк №{i}, score = {node.score:.3f}, объект = {node.metadata.get('object')}, год = {node.metadata.get('year')}"
        )
        st.text(node.node.get_content())
        st.markdown("---")
