# 0.1.0
> 2026年3月3日

> 跑通注册登录+简陋版本的RAG


# 0.1.1
> 混合检索+一个简单的对话接口
> 
> bm25索引放在modelSet下面， 
> 
> 本项目的所有与模型相关的都放在modelSet下面
> 
> 对parse.py进行分解，太庞大了（待做）
> 

# 问题
## 未解决
1. Fast API启动怎么这么慢
> 改成热加载试试

2. 我好像没有用到langchain，下面是我感觉可以用到的地方
- LLM 调用抽象，当前是直接用dashscope的
- prompt模版管理，就是用 prompt_template | llm 这种写法
- RAG 流程编排，这个最有用
```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 你的 retriever 已经准备好了
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 创建 RAG 链
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

response = rag_chain.invoke({"input": query})
```

- 文档加载和分块
> 这个不用换，因为规章制度的用我自己的就行，他默认的不咋地
>
> 优化: 我可以加入一个父索引，就是标题1>标题2>标题3，直接怼进去

3. 混合检索,还有几个网页没有入库，（查询改写,HyDE，同义词替换，同义词改写后续加) 
