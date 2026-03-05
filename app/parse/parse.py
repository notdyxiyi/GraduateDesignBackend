"""
第一次尝试
page 从0开始的哈
"""
import json
import re
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
import jieba
from sentence_transformers import SentenceTransformer
from dashscope import Generation
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import pickle
# 加载 .env 文件中的配置
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip('"')
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "").strip('"')

# 预定义的标签列表
TAGS_LIST = [
    "学生纪律", "学费减收", "勤工助学", "国家助学金", "励志奖学金",
    "专科国家奖学金", "政府奖学金", "大学生社会实践活动管理办法",
    "最低生活保障", "特困生免费主食", "王克昌奖学金", "暑期社会实践",
    "困难认定", "助学贷款管理办法", "励志奖学金管理办法"
]

# 全局模型变量
embedding_model = None
reranker_model = None
bm25_index = None
document_store = None

# 模型路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "modelSet")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
EMBEDDING_MODEL_PATH = os.path.join(MODEL_PATH, "bge-small-zh-v1.5")
RERANKER_MODEL_PATH = os.path.join(MODEL_PATH, "bge-ranker-base")
BM25_INDEX_PATH = os.path.join(MODEL_PATH, "bm25_index.pkl")

def load_models():
    """加载向量化模型和 reranker 模型"""
    global embedding_model, reranker_model

    print(f"正在加载嵌入模型：{EMBEDDING_MODEL_PATH}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    # 测试维度
    test_embedding = embedding_model.encode("测试文本", normalize_embeddings=True)
    print(f"✓ 嵌入模型加载完成，输出维度：{test_embedding.shape[0]}")

    print(f"正在加载重排序模型：{RERANKER_MODEL_PATH}")
    from sentence_transformers import CrossEncoder
    reranker_model = CrossEncoder(RERANKER_MODEL_PATH)

    print("模型加载完成")
# ====================== 基础数据结构 ======================

@dataclass
class PDFBox:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self):
        return [self.page, self.x0, self.y0, self.x1, self.y1]


class ParentDoc:
    """父文档：一章"""
    def __init__(self, title: str, metadata: Dict):
        self.content_lines: List[str] = [] # 章的内容
        self.title = title
        self.metadata = metadata
        self.children: List["ChildDoc"] = []


class ChildDoc:
    """子文档：一条（包含其下的（ 一 ）等内容）"""
    def __init__(self, title: str, metadata: Dict, coord_boxes: List[PDFBox]):
        self.title = title            # 第X条……
        self.metadata = metadata      # 章名等
        self.coord_boxes = coord_boxes  # 该条标题在 PDF 里的坐标框列表
        self.content_lines: List[str] = []  # 条款正文多行

    @property
    def content(self) -> str:
        return "\n".join(self.content_lines)


# ====================== 辅助正则与预处理 ======================

def normalize_space(text: str) -> str:
    """合并空白，解决 OCR / PDF 识别零散空格问题"""
    return re.sub(r"\s+", " ", text).strip()


def is_chapter(line: str) -> bool:
    # 第一章 / 第二章 ...
    return bool(re.match(r"^第[一二三四五六七八九十百零]+章", line))


def is_article(line: str) -> bool:
    # 第一条 / 第1条 ...
    return bool(re.match(r"^第[一二三四五六七八九十百零]+条", line)) or \
           bool(re.match(r"^第[0-9]+条", line))


def is_subitem(line: str) -> bool:
    # （一） / (1)
    return bool(re.match(r"^（[一二三四五六七八九十]+）", line)) or \
           bool(re.match(r"^$[0-9]+$", line))


def tag_chapter_with_ai(chapter_content: str) -> List[str]:
    TAGS_LIST.append("".join(chapter_content.split("\n")[1:]))
    """
    调用 Qwen AI 为章节内容打标签
    """
    prompt = f"""你是一个规章制度分类助手。请阅读以下章节内容，然后从给定的标签列表中选择最匹配的标签（可以多选）。

给定标签列表：{TAGS_LIST}

章节内容：
{chapter_content}

要求：
1. 只从上述给定标签中选择，不要创造新标签
2. 如果内容涉及多个标签，可以选择多个
3. 如果没有匹配的标签，返回空列表
4. 直接返回 JSON 格式的标签数组，例如：["国家助学金", "励志奖学金"]

请返回："""

    try:
        messages = [
            {'role': 'system', 'content': '你是一个专业的规章制度分类助手。'},
            {'role': 'user', 'content': prompt}
        ]

        response = Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qwen-plus",
            messages = messages,
            result_format="message"
        )

        if response.status_code == 200:
            ai_response = response.output.choices[0].message.content

            # 提取 JSON 部分
            match = re.search(r'\[.*?\]', ai_response, re.DOTALL)
            TAGS_LIST.pop()
            if match:
                tags = json.loads(match.group())
                return tags

            return []
        else:
            print(f"API 调用失败：{response.status_code}")
            print(f"错误码：{response.code}")
            print(f"错误信息：{response.message}")
            return []

    except Exception as e:
        print(f"打标签时出错：{e}")
        return []


# ====================== 坐标计算（关键词 → 坐标） ======================

def find_text_boxes(doc: fitz.Document, text: str) -> List[PDFBox]:
    """
    用 PyMuPDF 搜索文本，返回所有匹配的坐标框
    注意：这里直接用原文搜索；若 OCR 空格太乱，可再做“去空格匹配”扩展
    """
    boxes: List[PDFBox] = []
    if not text:
        return boxes

    for page_index in range(len(doc)):
        page = doc[page_index]
        # 直接按字符串搜索，返回 Rect 列表
        rects = page.search_for(text)
        for r in rects:
            boxes.append(PDFBox(page=page_index, x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1))
    return boxes


def get_embedding(text: str) -> List[float]:
    """使用本地 bge 模型获取文本向量"""
    if embedding_model is None:
        load_models()

    embedding = embedding_model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def save_to_chromadb(parents: List[Dict], children: List[Dict], collection_name: str = "regulations"):
    """
    将父子文档数据存储到 ChromaDB 向量数据库（使用本地 bge 模型）
    """
    global bm25_index, document_store
    # 加载模型
    load_models()

    # 初始化 ChromaDB 客户端（使用持久化存储 + 自定义函数）
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"创建新集合：{collection_name}")

    # 准备父文档数据
    parent_ids = []
    parent_documents = []
    parent_metadatas = []
    parent_embeddings = []

    for parent in parents:
        parent_ids.append(parent["id"])
        parent_documents.append(parent["content"])
        parent_metadatas.append({
            "type": "parent",
            "title": parent["title"],
            "chapter": parent["metadata"]["chapter"],
            "tags": ",".join(parent["metadata"].get("tags", [])),
            "page_start": str(parent["metadata"].get("page_start", 0)),
            "source_file": parent["metadata"].get("source_file", "unknown")  # 新增

        })
        # 使用本地模型计算向量
        parent_embeddings.append(get_embedding(parent["content"]))

    # 准备子文档数据
    child_ids = []
    child_documents = []
    child_metadatas = []
    child_embeddings = []

    # 创建 parent_id 到 source_file 的映射
    parent_source_map = {p["id"]: p["metadata"].get("source_file","unknow") for p in parents}
    import json
    for child in children:
        child_ids.append(child["id"])
        child_documents.append(child["content"])
        child_metadatas.append({
            "type": "child",
            "title": child["title"],
            "chapter": child["metadata"]["chapter"],
            "article": child["metadata"]["article"],
            "tags": ",".join(child["metadata"].get("tags", [])),
            "page": str(child["metadata"].get("page", 0)),
            "parent_id": child["parent_id"],
            "coords": json.dumps(child["metadata"].get("coords", [])),  # 转为 JSON 字符串存储
            "source_file":parent_source_map.get(child["parent_id"],"unknow")
        })
        # 使用本地模型计算向量
        child_embeddings.append(get_embedding(child["content"]))

    # 如果 collection 中已有数据，先删除
    if collection.count() > 0:
        existing = collection.get(include=[])
        if existing['ids']:
            collection.delete(ids=existing['ids'])

    # 添加父文档到 ChromaDB（使用预计算的向量）
    if parent_documents:
        collection.add(
            documents=parent_documents,
            metadatas=parent_metadatas,
            embeddings=parent_embeddings,
            ids=parent_ids
        )
        print(f"成功添加 {len(parent_documents)} 个父文档到 ChromaDB")

    # 添加子文档到 ChromaDB（使用预计算的向量）
    if child_documents:
        collection.add(
            documents=child_documents,
            metadatas= child_metadatas,
            embeddings=child_embeddings,
            ids=child_ids
        )
        print(f"成功添加 {len(child_documents)} 个子文档到 ChromaDB")
        # 构建 BM25 索引
        print("正在构建 BM25 索引...")
        all_docs = parent_documents + child_documents
        tokenized_docs = [list(jieba.cut(doc)) for doc in all_docs]
        bm25_index = BM25Okapi(tokenized_docs)
        document_store = {
            'documents': parent_documents + child_documents,
            'metadatas': parent_metadatas + child_metadatas,
            'ids': parent_ids + child_ids
        }
        print(f"BM25 索引构建完成，共 {len(tokenized_docs)} 个文档")
    # 保存 BM25 索引到文件
    save_bm25_index()
    print(f"ChromaDB 中总文档数：{collection.count()}")
    return collection


def query_with_rerank(collection, query_text: str, n_results: int = 5):
    """
    使用 bge-reranker 进行重排序的查询
    """
    if reranker_model is None:
        load_models()
    query_emb = embedding_model.encode(query_text, normalize_embeddings=True)

    # 先从 ChromaDB 获取候选结果（多取一些给 reranker 筛选）
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results * 3,  # 先取 3 倍数量
        include=["documents", "metadatas", "distances"]
    )

    if not results['documents'] or not results['documents'][0]:
        return results

    # 准备 rerank 的数据
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    # 构建 (query, document) 对用于 rerank
    pairs = [[query_text, doc] for doc in documents]

    # 使用 reranker 模型打分
    scores = reranker_model.predict(pairs)

    # 按分数排序
    ranked_indices = np.argsort(scores)[::-1][:n_results]

    # 构建最终结果
    final_results = {
        'documents': [[documents[i] for i in ranked_indices]],
        'metadatas': [[metadatas[i] for i in ranked_indices]],
        'distances': [[float(distances[i]) for i in ranked_indices]],
        'scores': [[float(scores[i]) for i in ranked_indices]]
    }

    return final_results


def query_with_hybrid_search(collection, query_text: str, n_results: int = 5, alpha: float = 0.7):
    """
    混合检索：结合向量检索和 BM25 检索
    alpha: 向量检索的权重 (0-1)，0.7 表示向量占 70%，BM25 占 30%
    """
    global bm25_index, document_store

    if reranker_model is None or embedding_model is None:
        load_models()
    # 检查 BM25 索引是否已加载，未加载则返回空结果
    if not load_bm25_index():
        print("⚠️ BM25 索引未初始化，请先运行 save_to_chromadb 函数")
        return {
            'documents': [[]],
            'metadatas': [[]],
            'scores': [[]]
        }
    # 1. 向量检索
    query_emb = embedding_model.encode(query_text, normalize_embeddings=True)
    vector_results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results * 3,
        include=["documents", "metadatas", "distances"]
    )

    # 2. BM25 检索
    query_tokens = list(jieba.cut(query_text))
    bm25_scores = bm25_index.get_scores(query_tokens)

    # 归一化 BM25 分数到 0-1
    if bm25_scores.max() > 0:
        bm25_scores_norm = bm25_scores / bm25_scores.max()
    else:
        bm25_scores_norm = bm25_scores

    # 3. 合并结果
    vector_docs = vector_results['documents'][0] if vector_results['documents'] else []
    vector_metas = vector_results['metadatas'][0] if vector_results['metadatas'] else []
    vector_distances = vector_results['distances'][0] if vector_results['distances'] else []

    # 将向量距离转换为相似度分数 (1 - distance)
    vector_scores = [1 - d for d in vector_distances]

    # 创建文档到索引的映射
    doc_to_idx = {doc: idx for idx, doc in enumerate(document_store['documents'])}

    # 4. 融合分数
    hybrid_results = {}
    for i, (doc, meta, v_score) in enumerate(zip(vector_docs, vector_metas, vector_scores)):
        doc_idx = doc_to_idx.get(doc, -1)
        b_score = bm25_scores_norm[doc_idx] if doc_idx >= 0 else 0

        # 加权融合
        hybrid_score = alpha * v_score + (1 - alpha) * b_score
        
        # 将最终得分加入 metadata
        meta_with_score = meta.copy()  # 避免修改原始数据
        meta_with_score['hybrid_score'] = hybrid_score
        meta_with_score['vector_score'] = v_score
        meta_with_score['bm25_score'] = b_score
        # 解析 coords 字符串为列表
        if 'coords' in meta_with_score and isinstance(meta_with_score['coords'], str):
            try:
                meta_with_score['coords'] = json.loads(meta_with_score['coords'])
            except:
                meta_with_score['coords'] = []
        
        hybrid_results[doc] = {
            'hybrid_score': hybrid_score,
            'metadata': meta_with_score,
            'vector_score': v_score,
            'bm25_score': b_score
        }

    # 5. 排序并返回 top-n
    sorted_results = sorted(hybrid_results.items(), key=lambda x: x[1]['hybrid_score'], reverse=True)[:n_results]

    final_results = {
        'documents': [[item[0] for item in sorted_results]],
        'metadatas': [[item[1]['metadata'] for item in sorted_results]],
        'hybrid_score': [[item[1]['hybrid_score'] for item in sorted_results]],
        'details': [{
            'document': item[0],
            'metadata': item[1]['metadata'],
            'hybrid_score': item[1]['hybrid_score'],
            'vector_score': item[1]['vector_score'],
            'bm25_score': item[1]['bm25_score']
        } for item in sorted_results]
    }

    return final_results



def get_embedding(text: str) -> List[float]:
    """使用本地 bge 模型获取文本向量"""
    if embedding_model is None:
        load_models()

    embedding = embedding_model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def save_to_chromadb_no_bm25(parents: List[Dict], children: List[Dict], collection_name: str = "regulations"):
    """
    将父子文档数据存储到 ChromaDB 向量数据库（使用本地 bge 模型）
    """
    # 加载模型
    load_models()

    # 初始化 ChromaDB 客户端（使用持久化存储 + 自定义函数）
    client = chromadb.PersistentClient(path="../chroma_db")

    # 获取或创建 collection
    # try:
    #     collection = client.get_collection(name=collection_name)
    #     print(f"使用已存在的集合：{collection_name}")
    # except:
    #     collection = client.create_collection(
    #         name=collection_name,
    #         metadata={"hnsw:space": "cosine"}
    #     )
    #     print(f"创建新集合：{collection_name}")
    try:
        client.delete_collection(name=collection_name)
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"创建新集合：{collection_name}")


    # 准备父文档数据
    parent_ids = []
    parent_documents = []
    parent_metadatas = []
    parent_embeddings = []

    for parent in parents:
        parent_ids.append(parent["id"])
        parent_documents.append(parent["content"])
        parent_metadatas.append({
            "type": "parent",
            "title": parent["title"],
            "chapter": parent["metadata"]["chapter"],
            "tags": ",".join(parent["metadata"].get("tags", [])),
            "page_start": str(parent["metadata"].get("page_start", 0)),
            "source_file": parent["metadata"].get("source_file", "unknown")  # 新增

        })
        # 使用本地模型计算向量
        parent_embeddings.append(get_embedding(parent["content"]))

    # 准备子文档数据
    child_ids = []
    child_documents = []
    child_metadatas = []
    child_embeddings = []

    for child in children:
        child_ids.append(child["id"])
        child_documents.append(child["content"])
        child_metadatas.append({
            "type": "child",
            "title": child["title"],
            "chapter": child["metadata"]["chapter"],
            "article": child["metadata"]["article"],
            "tags": ",".join(child["metadata"].get("tags", [])),
            "page": str(child["metadata"].get("page", 0)),
            "parent_id": child["parent_id"],
            "source_file": parents[child["parent_id"]]["metadata"].get("source_file", "unknown")  # 新增

        })
        # 使用本地模型计算向量
        child_embeddings.append(get_embedding(child["content"]))

    # 如果 collection 中已有数据，先删除
    if collection.count() > 0:
        existing = collection.get(include=[])
        if existing['ids']:
            collection.delete(ids=existing['ids'])

    # 添加父文档到 ChromaDB（使用预计算的向量）
    if parent_documents:
        collection.add(
            documents=parent_documents,
            metadatas=parent_metadatas,
            embeddings=parent_embeddings,
            ids=parent_ids
        )
        print(f"成功添加 {len(parent_documents)} 个父文档到 ChromaDB")

    # 添加子文档到 ChromaDB（使用预计算的向量）
    if child_documents:
        collection.add(
            documents=child_documents,
            metadatas=child_metadatas,
            embeddings=child_embeddings,
            ids=child_ids
        )
        print(f"成功添加 {len(child_documents)} 个子文档到 ChromaDB")

    print(f"ChromaDB 中总文档数：{collection.count()}")
    return collection




# ====================== 核心：从 PDF 构建父子索引 ======================

def build_parent_child_index_from_pdf(pdf_path: str,
                                      file_prefix: str = "") -> Tuple[List[Dict], List[Dict]]:
    """
    输入：规章制度 PDF 路径（纯文本、层级为 第一章 / 第一条 / （一））
    输出：父文档列表、子文档列表（已带坐标），可直接给前端或后续做 RAG
    参数:
        pdf_path: PDF 文件路径
        file_prefix: ID 前缀（用于区分不同 PDF 文件，避免 ID 重复）

    返回结构（示例）：
    parents = [
        {
            "id": "parent_0",
            "title": "第一章 总则",
            "metadata": {...}
        },
        ...
    ]

    children = [
        {
            "id": "child_0_0",
            "parent_id": "parent_0",
            "title": "第一条 ……",
            "content": "第一条 ……\n（一）……\n（二）……",

            "metadata": {..., "coords": [[page, x0, y0, x1, y1], ...]}
        },
        ...
    ]
    """
    doc = fitz.open(pdf_path)


    # 逐页抽行文本（不做复杂版面分析，只要“行文本 + 页码”）
    lines: List[Tuple[int, str]] = []  # (page_index, line_text)
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line.get("spans", []))
                text = normalize_space(text)
                if text:
                    lines.append((page_index, text))

    parents: List[ParentDoc] = []
    current_parent: ParentDoc = None
    current_child: ChildDoc = None

    for page_index, raw_line in lines:
        line = normalize_space(raw_line)

        # 识别“章”——新父文档
        if is_chapter(line):
            parent_meta = {
                "chapter": line,
                "page_start": page_index,
            }
            current_parent = ParentDoc(title=line, metadata=parent_meta)
            current_parent.content_lines.append(line)
            parents.append(current_parent)

            current_child = None
            continue

        # 识别“条”——新子文档
        if is_article(line) and current_parent is not None:
            # 用原始行文本去 PDF 搜索坐标（用于前端高亮条款标题）
            coord_boxes = find_text_boxes(doc, raw_line)

            child_meta = {
                "chapter": "\n".join(current_parent.content_lines),  # 与对应 parent 的 title 一致
                "article": line,
                "page": page_index,
            }
            current_child = ChildDoc(title=line, metadata=child_meta, coord_boxes=coord_boxes)
            # 第一行标题本身也放进内容里，方便检索
            current_child.content_lines.append(line)

            current_parent.children.append(current_child)
            continue

        # 识别“（一） / (1)”——仍算这条下面的内容
        if is_subitem(line) and current_child is not None:
            current_child.content_lines.append(line)
            continue

        # 其他正文行，继续挂在当前条之下
        if current_child is not None:
            current_child.content_lines.append(line)
        elif current_parent is not None:
            current_parent.content_lines.append(line)
    # 序列化成纯 dict，方便你直接存库 / 返回 / 喂给向量库
    parent_dicts: List[Dict] = []
    child_dicts: List[Dict] = []

    for p_idx, parent in enumerate(parents):
        parent_id = f"{file_prefix}parent_{p_idx}"
        # 为当前章调用 AI 打标签
        chapter_full_content = "\n".join(parent.content_lines)
        tags = tag_chapter_with_ai(chapter_full_content)

        parent_dicts.append({
            "id": parent_id,
            "content":"\n".join(parent.content_lines),
            "title":  "\n".join(parent.content_lines),
            "metadata": {
                **parent.metadata,
                "chapter": "\n".join(parent.content_lines),  # chapter 也改为完整内容
                "source_file": os.path.join(PROJECT_ROOT,os.path.basename(pdf_path)),
                "tags": tags  # ← 新增

            },
        })

        for c_idx, child in enumerate(parent.children):
            child_id = f"{file_prefix}child_{p_idx}_{c_idx}"
            child_dicts.append({
                "id": child_id,
                "parent_id": parent_id,
                "title": child.title,
                "content": child.content,
                "metadata": {
                    **child.metadata,
                    # 给前端：一个条款标题可能跨多行/多处，这里返回所有框
                    "coords": [box.to_list() for box in child.coord_boxes],
                    "tags": tags,  # ← 新增
                    "source_file": os.path.join(PROJECT_ROOT,os.path.basename(pdf_path)) # 新增：来源文件
                }
            })

    return parent_dicts, child_dicts


def test_ranker():
    # 测试查询（带 rerank）
    print("\n测试查询（使用 reranker 重排序）...")
    query_text = "国家助学金申请条件"
    results = query_with_rerank(collection, query_text, n_results=3)

    print(f"\n查询：{query_text}")
    print(f"查询结果：{json.dumps(results, ensure_ascii=False, indent=2)}")

def test_hybrid():
    print("\n测试查询（使用混合检索）...")
    query_text = "国家助学金申请条件"
    results = query_with_hybrid_search(collection, query_text, n_results=3, alpha=0.7)

    print(f"\n查询：{query_text}")
    print(f"查询结果:")
    for i, detail in enumerate(results['details'], 1):
        print(f"\n[{i}] 综合得分：{detail['hybrid_score']:.4f}")
        print(f"   向量得分：{detail['vector_score']:.4f}")
        print(f"   BM25 得分：{detail['bm25_score']:.4f}")
        print(f"   内容：{detail['document'][:100]}...")


def load_bm25_index():
    """加载 BM25 索引（如果存在）"""
    global bm25_index, document_store

    if os.path.exists(BM25_INDEX_PATH):
        print(f"正在加载 BM25 索引：{BM25_INDEX_PATH}")
        with open(BM25_INDEX_PATH, 'rb') as f:
            data = pickle.load(f)
            bm25_index = data['bm25_index']
            document_store = data['document_store']
        print(f"✓ BM25 索引加载完成，共 {len(document_store['documents'])} 个文档")
        return True
    else:
        print("⚠️ BM25 索引文件不存在，需要先构建")
        return False


def save_bm25_index():
    """保存 BM25 索引"""
    global bm25_index, document_store

    if bm25_index is not None and document_store is not None:
        print(f"正在保存 BM25 索引到：{BM25_INDEX_PATH}")
        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump({
                'bm25_index': bm25_index,
                'document_store': document_store
            }, f)
        print(f"✓ BM25 索引保存完成")
    else:
        print("⚠️ 没有可保存的 BM25 索引")


def load_build_index():
    import glob

    # 获取 files 目录下所有 PDF 文件
    pdf_dir = "../files"
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    if not pdf_files:
        print(f"❌ 在 '{pdf_dir}' 目录下未找到 PDF 文件")
        exit(1)

    print(f"📚 发现 {len(pdf_files)} 个 PDF 文件\n")

    # 遍历所有 PDF 构建索引
    all_parents = []
    all_children = []

    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        print(f"[{i}/{len(pdf_files)}] 正在处理：{filename}")
        # 生成文件前缀（去掉 .pdf 后缀，替换特殊字符）
        file_prefix = filename.replace(".pdf", "").replace(" ", "_").replace(".", "_") + "_"

        try:
            parents, children = build_parent_child_index_from_pdf(pdf_path,file_prefix=file_prefix)
            all_parents.extend(parents)
            all_children.extend(children)
            print(f"   ✅ 提取 {len(parents)} 个父文档，{len(children)} 个子文档\n")
        except Exception as e:
            print(f"   ❌ 处理失败：{e}\n")

    print(f"\n{'=' * 60}")
    print(f"📊 汇总：共 {len(all_parents)} 个父文档，{len(all_children)} 个子文档")
    print(f"{'=' * 60}\n")

    # 保存 JSON 输出
    output_data = {
        "parents": all_parents,
        "children": all_children
    }

    # 存储到 ChromaDB
    print("开始存储到 ChromaDB...")
    collection = save_to_chromadb(all_parents, all_children, collection_name="regulations")

    # 测试查询（带 rerank）
    print("\n测试查询（使用混合检索）...")
    query_text = "国家助学金申请条件"
    results = query_with_hybrid_search(collection, query_text, n_results=3, alpha=0.7)

    print(f"\n查询：{query_text}")
    print(f"查询结果:")
    for i, detail in enumerate(results['details'], 1):
        print(f"\n[{i}] 综合得分：{detail['hybrid_score']:.4f}")
        print(f"   向量得分：{detail['vector_score']:.4f}")
        print(f"   BM25 得分：{detail['bm25_score']:.4f}")
        print(f"   内容：{detail['document'][:100]}...")


def only_query():
    # 测试查询（带 rerank）
    print("\n测试查询（使用混合检索）...")
    query_text = "国家助学金申请条件"

    # 从现有 ChromaDB 获取 collection
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name="regulations")

    results = query_with_hybrid_search(collection, query_text, n_results=3, alpha=0.7)

    print(f"\n查询：{query_text}")
    print(f"查询结果:")
    for i, detail in enumerate(results['details'], 1):
        print(f"\n[{i}] 综合得分：{detail['hybrid_score']:.4f}")
        print(f"   向量得分：{detail['vector_score']:.4f}")
        print(f"   BM25 得分：{detail['bm25_score']:.4f}")
        print(f"   内容：{detail['document'][:100]}...")

# ====================== 简单示例调用 ======================
if __name__ == "__main__":
    load_build_index()