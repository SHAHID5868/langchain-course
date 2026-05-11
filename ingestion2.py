from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings





load_dotenv()

urls =[
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap=0)
doc_splts = text_splitter.split_documents(docs_list)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

PineconeVectorStore.from_documents( doc_splts,embedding=embedding, index_name="advanced-rag")

vector_store = PineconeVectorStore(embedding, index_name="advanced-rag")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})