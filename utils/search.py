import requests,aiohttp
from langchain_core.documents import Document
from configs.settings import SEARCH
from typing import List, Dict


class SearxngClient:
    def __init__(self, base_url=SEARCH["url"], timeout=SEARCH["time_out"]):
        self.base_url = base_url + "/search"
        self.timeout = timeout
        self.headers = {"Accept": "application/json"}
        # 验证Searxng连接
        self.validate_searxng_connection()

    def validate_searxng_connection(self):
        """验证Searxng连接"""
        test_query = "今天的头条新闻"
        try:
            results = self.search(test_query)
            if len(results) > 0:
                print("✅ Searxng连接正常")
                print(f"搜索 {test_query}")
                print(f"测试结果示例：{results[0]}")
            else:
                print("⚠️ 连接成功但无结果返回，请检查服务器配置")
        except Exception as e:
            print(f"❌ Searxng连接失败: {str(e)}")

    # 在 search 方法中修改 engines 默认值
    def search(self, query: str,
               categories: List[str] = ["general"],
               engines: List[str] = SEARCH["engine"],
               page: int = 1) -> List[Dict]:

        params = {
            "q": query,
            "categories": ",".join(categories),
            "engines": ",".join(engines),
            "pageno": page,
            "format": "json"
        }

        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["results"]
        except Exception as e:
            print(f"⚠️ Searxng搜索失败: {str(e)}")
            return []

    # 将搜索结果转换成 langchain 的 Document 格式
    def to_documents(self, results: List[Dict], min_score: float = 0.7) -> List[Document]:
        valid_docs = []
        for result in results:
            # ==== 元数据强制验证 ====
            if "source" not in result:
                result["source"] = "web"

            metadata = {
                "source": result.get("source", "web"),
                "url": result.get("url", "N/A"),  # 确保url字段存在
                "engine": result.get("engine", "未知引擎"),
                "title": result.get("title", "无标题")
            }

            content = f"标题：{result.get('title', '')}\n内容：{result.get('content', '')}"
            valid_docs.append(Document(
                page_content=content,
                metadata=metadata
            ))
        return valid_docs
