# Museum Opinion RAG

A Retrieval-Augmented Generation (RAG) application for museum-related opinions using LlamaIndex, ChromaDB, and Streamlit.
The app is intended for using it to analyze Russian reviews (the data is provided in the megatitan table), but can be used with other languages (embedding model used is **paraphrase-multilingual-MiniLM-L12-v2**).
The functionality allows the use of Groq API or local models with llama.cpp.

## Pipeline

<img width="361" height="802" alt="project_diagram" src="https://github.com/user-attachments/assets/c4743709-af1c-4e11-a1e5-4cb68566e9b7" />


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
```

## Using the app
1. Enter the name of the collection (default is 'sentiment_rag')
if the collection exists, you will jump to the next step immediately
if the collection is new, it will take some time to generate embeddings (not too long though)
2. Choose the inference engine (Groq or llama.cpp)
- with Groq, you will need to enter the name of the model and ensure .env file is in your folder
- with llama.cpp, you will need to provide the path to the model file
3. Enter your query
You will first see the answer of the model (with markdown if it so wishes) and then 15 relevant chunks with score, object and year info
