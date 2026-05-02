from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore



load_dotenv()






if __name__ == "__main__":
    print("Hello, World!")
    loader = TextLoader("/Users/shahid/PyCharmMiscProject/LangChain-course/mediumblog1.txt") # if there encoding isssue use encoding='utf-8' or auto_detect_encoding=True
    document = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("ingesting data into pinecone")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ.get("INDEX_NAME"),
    )