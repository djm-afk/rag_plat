from langchain_chroma import Chroma
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
            self.processor.original_metadata = raw_docs[0].metadata  # 传递元数据
            split_docs = self.processor.split_document(raw_docs[0].page_content)
            print("🔄 正在入库...")
            return Chroma.from_documents(
                documents=split_docs,
                embedding=self.embedding_model,
                persist_directory=str(PATHS["vector_db"]),
                collection_metadata={"hnsw:space": "cosine"}  # 明确指定相似度计算方式
            )
        else:
            print("🔄 加载已有数据库")
            return Chroma(
                persist_directory=str(PATHS["vector_db"]),
                embedding_function=self.embedding_model,
                collection_metadata={"hnsw:space": "cosine"}  # 明确指定相似度计算方式
            )