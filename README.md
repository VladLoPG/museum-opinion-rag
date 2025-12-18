# Museum Opinion RAG

A Retrieval-Augmented Generation (RAG) application for museum-related opinions using LlamaIndex, ChromaDB, and Streamlit.

## Getting Started

### 1 Clone the repository and enter the folder

```bash
git clone https://github.com/VladLoPG/museum-opinion-rag.git
cd museum-opinion-rag/llama_index_rag
``` 

### 2 Activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 3 Install requirements (There are many, but they are needed. See requirements.in for intended packages)
```bash
pip install -r requirements.txt
```

### 4 Configure the environment variable (if you are using Groq) and/or use a local GGUF model
```bash
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### 5 Run the app (the app should open automatically or see the terminal for the manual link)
```bash
streamlit run app.py

