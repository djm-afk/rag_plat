from core.vector_db import VectorDatabase
from core.qa_system import QASystemBuilder
from utils.search import SearxngClient


def main():
    # 加载搜索引擎
    searxng_client = SearxngClient()

    # 加载数据库
    vector_db = VectorDatabase().vector_store

    # 问答系统构建
    qa_system = QASystemBuilder(
        vector_db=vector_db,
        searxng_client=searxng_client,
        enable_web=True
    ).create_chain()

    # 执行查询
    questions = [
        "2025年3月6号广州的天气？",
        "2025年有什么新闻？"
    ]

    for question in questions:
        """增强版流式响应"""
        print("问题：", question)
        print("答案生成：", end="")

        # 执行问答链
        result = qa_system.invoke({"query": question})

        # 显示来源文档
        print("\n--------来源文档详情--------")
        for doc in result['source_documents']:
            source = doc.metadata.get("source", "本地知识库")
            if source == "web":
                print(f"🌐 [{doc.metadata.get('engine')}] {doc.metadata.get('url')}")
                print(f"   {doc.page_content[:100]}...")
            else:
                print(f"📚 [{doc.metadata.get('Header2', '无标题')}]")
                print(f"   {doc.page_content[:100]}...")
        print("\n--------------------------")

if __name__ == "__main__":
    main()