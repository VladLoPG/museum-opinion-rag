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
    st.session_state.query_engine = set_llm(index)
    st.session_state.collection_loaded = True

if st.session_state.collection_loaded:
    st.divider()
    query = st.text_input(label="Введите запрос для поиска по базе")

    if st.button("Начать поиск"):
        get_response(query_engine=st.session_state.query_engine, query=query)
