import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.storage_context import StorageContext
from config import OPENAI_API_KEY

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in config.py")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def create_index_from_text(text_dict):
    with open("temp_doc.txt", "w", encoding="utf-8") as f:
        for page, text in text_dict.items():
            f.write(f"\n\n--- Page {page} ---\n{text}")

    documents = SimpleDirectoryReader(input_files=["temp_doc.txt"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

def chat_with_index(index, query):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return str(response)
