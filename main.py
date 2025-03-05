from core.vector_db import VectorDatabase
from core.qa_system import QASystemBuilder


def main():
    # 加载数据库
    vector_db = VectorDatabase().vector_store

    # 问答系统构建
    qa_system = QASystemBuilder(vector_db).create_chain()

    # 执行查询
    questions = [
        "藜麦有哪些营养价值",
        "西藏种植藜麦需要哪些环境条件",
        "如何用藜麦制作食品"
    ]

    for question in questions:
        print("问题：", question)
        print("答案生成：", end="")
        response = qa_system.invoke({"query": question})
        if 'source_documents' in response:
            print("\n--------来源文档如下--------")
            for doc in response['source_documents']:
                print(f" - {doc.metadata.get('Header2', '无标题')}: {doc.page_content[:50]}...")
        else:
            print("\n未找到相关来源文档")


if __name__ == "__main__":
    main()