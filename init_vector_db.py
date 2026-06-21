from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# 加载环境密钥
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# 1.加载知识库全部txt文档
doc_paths = [
    "./knowledge_base/fraud_case.txt",
    "./knowledge_base/finance_know.txt"
]
documents = []
for path in doc_paths:
    loader = TextLoader(path, encoding="utf-8")
    documents.extend(loader.load())

# 2.文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
split_docs = text_splitter.split_documents(documents)

# 3.初始化阿里嵌入模型
embedding = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=api_key
)

# 4.生成向量库，持久化保存到chroma_db文件夹
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory="./chroma_db"
)
vectordb.persist()
print("知识库向量库构建完成！")