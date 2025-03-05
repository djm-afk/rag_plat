from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from core.document_loader import DocumentLoader, MarkdownProcessor
from configs.settings import PATHS, MODELS


class VectorDatabase:
    """向量数据库管理器"""

    def __init__(self):
        # 初始化组件
        self.loader = DocumentLoader()
        self.processor = MarkdownProcessor()
        self.embedding_model = self._init_embedding()
        self.vector_store = self._init_vector_db()

    def _init_embedding(self) -> HuggingFaceEmbeddings:
        print("🔄 初始化嵌入模型...")
        """初始化嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name=MODELS["embedding"]["name"],
            model_kwargs={"device": MODELS["embedding"]["device"]},
            encode_kwargs=MODELS["embedding"]["encode_kwargs"]
        )

    def _init_vector_db(self) -> Chroma:
        print("🔄 加载向量数据库...")
        """初始化/加载向量数据库"""
        if not PATHS["vector_db"].exists() or not list(PATHS["vector_db"].iterdir()):
            print("⚠️ 检测到空数据库 - 正在创建向量库...")
            # 文档处理
            raw_docs = self.loader.load_with_fallback()
            split_docs = self.processor.split_document(raw_docs[0].page_content)
            print("🔄 正在入库...")
            return Chroma.from_documents(
                documents=split_docs,
                embedding=self.embedding_model,
                persist_directory=str(PATHS["vector_db"]),  # 指定持久化目录
            )
        else:
            print("🔄 加载已有数据库")
            return Chroma(
                persist_directory=str(PATHS["vector_db"]),
                embedding_function=self.embedding_model
            )

    def add_documents(self, docs: List[Document], force_rebuild: bool = False) -> None:
        """
        文档入库核心方法
        :param docs: 预处理后的文档列表
        :param force_rebuild: 是否强制重建库
        """
        try:
            if force_rebuild and PATHS["vector_db"].exists():
                import shutil
                shutil.rmtree(PATHS["vector_db"])
                print("♻️ 已清除旧数据库")

            # 自动创建或重建数据库
            self.vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                persist_directory=str(PATHS["vector_db"])
            )
            print(f"✅ 成功入库 {len(docs)} 个文档块")

            # 添加后续优化
            # self._post_processing()

        except Exception as e:
            print(f"❌ 入库失败: {str(e)}")
            raise RuntimeError("文档入库流程异常") from e

    def _post_processing(self):
        """入库后优化操作"""
        # 示例：手动触发持久化
        self.vector_store.persist()
        # 可扩展添加索引优化等操作