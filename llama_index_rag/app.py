from tools import (
    load_csv,
    create_json,
    chunk_dataset,
    get_documents,
    set_vector_store,
    get_vector_store,
    set_llm,
    get_response,
)
import chromadb
import streamlit as st

if "index" not in st.session_state:
    st.session_state.index = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "collection_exists" not in st.session_state:
    st.session_state.collection_exists = False
if "collection_loaded" not in st.session_state:
    st.session_state.collection_loaded = False
if "inference" not in st.session_state:
    st.session_state.inference = None
if "model" not in st.session_state:
    st.session_state.model = None


collection_name = st.text_input(
    label="Введите название коллекции или оставьте без изменений", value="sentiment_rag"
)

chroma_client = chromadb.PersistentClient(path="./chroma")
existing = [c.name for c in chroma_client.list_collections()]
st.session_state.collection_exists = collection_name in existing

if not st.session_state.collection_exists and not st.session_state.collection_loaded:
    path = st.text_input(
        label="Введите путь к файлу с данными отзывов",
        value="megatitan_texts_unique.csv",
    )
else:
    path = None

if st.button("Начать загрузку коллекции"):
    if not st.session_state.collection_exists:
        data = load_csv(path=path)
        json_data = create_json(data)
        chunked_data = chunk_dataset(json_data)
        documents = get_documents(chunked_data)
        index = set_vector_store(documents, collection_name=collection_name)

    else:
        index = get_vector_store(collection_name=collection_name)

    st.session_state.index = index
    st.session_state.collection_loaded = True
    
if st.session_state.collection_loaded:    
    inference = st.selectbox(label='Выберите движок для инференса', options=['Groq', 'llamacpp'])
    st.session_state.inference = inference
    if st.session_state.inference == 'Groq':
        model = st.text_input(label='Введите название модели (по умолчанию можно взять - llama-3.1-8b-instant)')
    else:
        model = st.text_input(label='Введите путь к модели GGUF')
    st.session_state.model = model

if st.session_state.inference and st.session_state.model:
    query_engine = set_llm(_index=st.session_state.index, inference=st.session_state.inference, model=st.session_state.model)  
    st.session_state.query_engine = query_engine
    

if st.session_state.query_engine:
    query = st.text_input(label="Введите запрос для поиска по базе")

    if st.button("Начать поиск"):
        st.text('Поиск запущен')
        get_response(query_engine=st.session_state.query_engine, query=query)
