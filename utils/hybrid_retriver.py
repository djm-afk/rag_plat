from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from pydantic import Field
from utils.search import SearxngClient
from configs.settings import SEARCH


class HybridRetriever(BaseRetriever):
    """支持本地+网络搜索的混合检索器"""
    vector_db: Chroma = Field(...)
    searxng_client: SearxngClient = Field(...)
    enable_web: bool = Field(default=True)
    top_k: int = Field(default=SEARCH["top_k"])

    # Chroma
    def _get_local_docs(self, query: str) -> List[Document]:
        # 新版 Chroma 的正确调用方式
        print(f"🔍 知识库检索")
        docs_with_scores = self.vector_db.similarity_search_with_score(
            query=query,
            k=5,
        )
        # 设置相似度阈值
        similarity_threshold = 0.3  # 相似度分数阈值（分数越低表示越相似）

        # 双重过滤：分数阈值 + 最大返回数量
        filtered_docs = [doc for doc, score in docs_with_scores if score <= similarity_threshold][:3]  # 最终返回 top_k 个

        return filtered_docs

    # Elasticsearch
    def _get_es_local_docs(self, query: str) -> List[Document]:
        # Elasticsearch 相似度查询
        search_body = {
            "knn": {
                "field": "vector",
                "query_vector": self.vector_store.embedding.embed_query(query),
                "k": 5,
                "num_candidates": 50
            },
            "fields": ["content", "metadata"]
        }

        response = self.vector_store.client.search(
            index=self.vector_store.index_name,
            body=search_body
        )

        return [
            Document(
                page_content=hit["_source"]["content"],
                metadata=hit["_source"].get("metadata", {})
            ) for hit in response["hits"]["hits"]
        ]

    def _get_web_docs(self, query: str) -> List[Document]:
        """网络搜索检索"""
        if not self.enable_web:
            return []

        try:
            print(f"🔍 网络搜索")
            results = self.searxng_client.search(query)
            return self.searxng_client.to_documents(results)[:self.top_k]
        except Exception as e:
            print(f"⚠️ 网络搜索失败: {str(e)}")
            return []

    def _merge_results(self, local_docs: List[Document], web_docs: List[Document]) -> List[Document]:
        # 强制元数据补全
        for doc in local_docs + web_docs:
            doc.metadata.setdefault("source", "local" if doc in local_docs else "web")
            doc.metadata.setdefault("url", "N/A")

        # 优先保留网络结果，并按相关性排序
        combined = sorted(
            web_docs + local_docs,
            key=lambda x: (
                -x.metadata.get("rank", 0) if x.metadata["source"] == "web" else 0,
                len(x.page_content)
            ),
            reverse=True
        )

        # 基于标题和首段内容去重
        seen = set()
        merged = []
        for doc in combined:
            identifier = f"{doc.metadata.get('title', '')[:30]}_{hash(doc.page_content[:200])}"
            if identifier not in seen:
                seen.add(identifier)
                merged.append(doc)
        return merged[:12]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """核心检索逻辑"""
        # 并行执行本地和网络检索
        local_docs = self._get_local_docs(query)
        web_docs = self._get_web_docs(query)

        # 合并结果
        return self._merge_results(local_docs, web_docs)