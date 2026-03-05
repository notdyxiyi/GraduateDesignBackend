"""
对话相关
"""

"""
对话相关
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field
import json
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.parse.parse import query_with_hybrid_search, load_models, embedding_model, bm25_index
from app.database import get_db
from sqlalchemy.orm import Session

router = APIRouter(tags=["dialog"])

# 1. 定义流式请求模型 Swagger
class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", example="国家助学金申请条件是什么？")
    n_results: int = Field(default=3, description="返回结果数量", ge=1, le=10)
    alpha: float = Field(default=0.7, description="向量检索权重 (0-1)", ge=0.0, le=1.0)

# 2. 定义非流式响应模型 Swagger
class ChatResponse(BaseModel):
    question: str = Field(
        description="用户输入的问题",
        examples = ["奖学金的条件","挂科有什么后果?"]
    )
    answer: str = Field(
        description="生成的答案",
        examples = ["奖学金的条件","挂科有什么后果?"]
    )
    sources: list[dict] = Field(
        description="参考来源",
        examples = [
            {
                "index": 1,
                "source_file": "1.天津职业技术师范大学天津市人民政府奖学金评审及管理办法.pdf",
            }
        ]
    )

async def generate_stream(question: str, n_results: int, alpha: float):
    """生成器函数，用于流式输出"""
    try:
        # 加载模型（如果还没加载）
        if embedding_model is None or bm25_index is None:
            load_models()

        # 从数据库获取 collection
        db = next(get_db())
        from app.parse.parse import save_to_chromadb

        # 这里需要传入 collection，实际应该从全局状态或数据库获取
        # 简化处理：假设已经初始化过
        import chromadb
        client = chromadb.PersistentClient(path=os.path.join(project_root, "chroma_db"))
        collection = client.get_collection(name="regulations")

        # 执行混合检索
        results = query_with_hybrid_search(collection, question, n_results=n_results, alpha=alpha)

        # 构造回答
        answer_parts = []
        for i, detail in enumerate(results['details'], 1):
            source = detail['metadata'].get('source_file', '未知')
            article = detail['metadata'].get('article', '')
            content = detail['document']

            part = f"【参考{i}】{source} {article}\n{content}\n\n"
            answer_parts.append(part)

        # 流式输出每个部分
        response_data = {
            "question": question,
            "answer": "正在生成回答...",
            "sources": []
        }

        yield f"data: {json.dumps({'type': 'start', 'data': response_data}, ensure_ascii=False)}\n\n"

        for i, part in enumerate(answer_parts, 1):
            chunk = {
                "type": "chunk",
                "data": {
                    "index": i,
                    "content": part
                }
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # 完成
        final_data = {
            "type": "end",
            "data": {
                "total_sources": len(results['details']),
                "message": "回答完成"
            }
        }
        yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_data = {
            "type": "error",
            "data": {
                "error": str(e)
            }
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


@router.post("/query",
             summary = "流式问答接口")
async def query_dialog(request: QueryRequest):
    """
    流式问答接口
    """
    return StreamingResponse(
        generate_stream(request.question, request.n_results, request.alpha),
        media_type="text/event-stream"
    )


@router.post("/chat",
             summary="普通问答接口")
async def chat_simple(request: QueryRequest):
    """
    简单问答接口（非流式，一次性返回）
    """
    try:
        # 加载模型
        if embedding_model is None or bm25_index is None:
            load_models()

        # 获取 collection
        import chromadb
        client = chromadb.PersistentClient(path=os.path.join(project_root, "chroma_db"))
        collection = client.get_collection(name="regulations")

        # 执行混合检索
        results = query_with_hybrid_search(collection, request.question, n_results=request.n_results,
                                           alpha=request.alpha)

        # 构造回答
        sources = []
        for i, detail in enumerate(results['details'], 1):
            sources.append({
                "index": i,
                "source_file": detail['metadata'].get('source_file', '未知'),
                "article": detail['metadata'].get('article', ''),
                "content": detail['document'],
                "score": detail['hybrid_score']
            })

        return {
            "question": request.question,
            "sources": sources,
            "total": len(sources)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
