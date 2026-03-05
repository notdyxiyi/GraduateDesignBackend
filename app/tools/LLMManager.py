"""
LLM 工具类 - 使用 LangChain 管理大语言模型
"""
import os
from dotenv import load_dotenv
from langchain_community.llms import Tongyi
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
load_dotenv()


class RAGLLM:
    """RAG 专用的 LLM 管理类"""

    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL", "qwen-plus")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        self.api_key = os.getenv("DASHSCOPE_API_KEY")

        # 初始化 LLM
        self.llm = Tongyi(
            model=self.model_name,
            dashscope_api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # 加载提示词模板
        from app.prompts.prompts import AFTER_RANKER_PROMPT
        self.prompt_template = PromptTemplate(
            template=AFTER_RANKER_PROMPT,
            input_variables=["question", "context"]
        )

        # 创建 LLM 链
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def generate_answer(self, question: str, chunks: list[dict]) -> dict:
        """
        基于检索到的 chunks 生成回答

        Args:
            question: 用户问题
            chunks: 经过ranker 排序后的 chunk 列表，每个包含 content 和 metadata

        Returns:
            LLM 生成的回答
        """
        # 构造 context，给每个 chunk 编号
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get('document', '')
            source = chunk.get('metadata', {}).get('source_file', '未知')
            article = chunk.get('metadata', {}).get('article', '')

            context_parts.append(f"[{i}] {source} {article}\n{content}")

        context = "\n\n".join(context_parts)

        response = self.chain.invoke({
            "question": question,
            "context": context
        })
        llm_response_with_id = response["text"]
        # 去掉编号，返回一个列表
        # 存放引用的列表
        references_list = []
        # 利用正则对llm_response_with_id提取出来[1][2][3]这样的
        citation_matches = re.findall(r'\[(\d+)\]', llm_response_with_id)

        # 去重并转换为整数索引
        # 方框里面的数字-1，如[1] 即1-1=0，那么就是chunks所对应的索引，引用是在chunks[i]的 metadata当中，直接返回即可
        cited_indices = sorted(set(int(num) - 1 for num in citation_matches))
        # 从 chunks 中获取对应的引用信息
        for idx in cited_indices:
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                references_list.append({
                    "index": idx + 1,  # 引用编号（从 1 开始）
                    "source_file": chunk["metadata"].get("source_file", "未知"),
                    "article": chunk["metadata"].get("article", ""),
                    "content": chunk["document"],
                    "coords": chunk["metadata"].get("coords", [])
                })

        # 返回结果
        res = {
            "answer": llm_response_with_id,
            "references": references_list
        }

        return res



# 全局单例
_llm_instance = None


def get_llm() -> RAGLLM:
    """获取 LLM 实例（单例模式）"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = RAGLLM()
    return _llm_instance
