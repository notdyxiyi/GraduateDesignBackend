"""
简单的 RAG 系统 - 使用 LangChain + Qwen 大模型
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.llms import Tongyi
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# 加载环境变量
load_dotenv()

# 开启离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
# 1. 准备文档数据（示例文本）
documents = [
    Document(
        page_content="""
        FastAPI 是一个现代、快速（高性能）的 Web 框架，用于使用 Python 3.7+ 构建 API。
        它基于标准 Python 类型提示，具有自动交互式文档（Swagger UI），
        并且是异步支持的。FastAPI 由 Sebastián Ramírez 创建，已成为构建 
        RESTful API 的最流行框架之一。
        """,
        metadata={"source": "fastapi_intro"}
    ),
    Document(
        page_content="""
        LangChain 是一个用于开发语言模型应用的框架。它提供了工具和抽象来简化
        整个应用程序的开发过程，从数据收集到推理再到评估。LangChain 支持多种
        大语言模型提供商，包括 OpenAI、Anthropic、Hugging Face 等。
        """,
        metadata={"source": "langchain_intro"}
    ),
    Document(
        page_content="""
        RAG（检索增强生成）是一种 AI 技术，结合了信息检索和文本生成的优势。
        它首先从知识库中检索相关文档，然后将这些文档作为上下文提供给大语言模型，
        让模型基于检索到的信息生成更准确、更有根据的回答。RAG 可以有效减少
        大模型的幻觉问题。
        """,
        metadata={"source": "rag_intro"}
    ),
    Document(
        page_content="""
        向量数据库是一种专门用于存储和查询向量数据的数据库系统。与传统的关系型
        数据库不同，向量数据库使用向量嵌入（embeddings）来表示数据，这使得它们
        非常适合语义搜索、相似度匹配等任务。常见的向量数据库包括 FAISS、
        Chroma、Pinecone 等。
        """,
        metadata={"source": "vector_db_intro"}
    )
]

# 2. 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

texts = text_splitter.split_documents(documents)


# 设置缓存目录到项目根目录
cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "embeddings")
os.makedirs(cache_dir, exist_ok=True)


# 3. 创建 embeddings（使用本地模型）
print("正在加载 embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="../modelSet/all-MiniLM-L6V2",
    model_kwargs={'device': 'cpu','local_files_only':True},
    encode_kwargs={'normalize_embeddings': True},
    cache_folder= cache_dir
)

# 4. 创建向量数据库
print("正在构建向量索引...")
vectorstore = FAISS.from_documents(texts, embeddings)

# 5. 初始化 Qwen 大模型
print("正在初始化 Qwen 大模型...")
llm = Tongyi(
    model="qwen-turbo",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.7
)

# 6. 创建 RAG 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

def ask_question(question: str):
    """
    提问函数
    
    Args:
        question: 用户的问题
        
    Returns:
        回答和来源文档
    """
    print(f"\n{'='*60}")
    print(f"问题：{question}")
    print(f"{'='*60}")
    
    result = qa_chain.invoke({"query": question})
    
    print(f"\n回答：{result['result']}")
    print(f"\n参考文档:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\n[文档 {i}] 来源：{doc.metadata.get('source', 'unknown')}")
        print(f"内容：{doc.page_content[:200]}...")
    
    return result

if __name__ == "__main__":
    print("="*60)
    print("简易 RAG 问答系统".center(60))
    print("="*60)
    print("\n已加载以下文档:")
    for doc in documents:
        print(f"  - {doc.metadata['source']}")
    
    # 测试问题
    questions = [
        "什么是 FastAPI？",
        "RAG 是什么？",
        "向量数据库有什么特点？"
    ]
    
    for q in questions:
        ask_question(q)
        input("\n按回车继续...")
    
    print("\n测试完成！现在你可以自由提问了。")
