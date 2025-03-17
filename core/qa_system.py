from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai.chat_models import ChatOpenAI
from utils.hybrid_retriver import HybridRetriever
from configs.settings import MODELS,SEARCH


class QASystemBuilder:
    """问答系统构造器"""

    def __init__(self, vector_db, searxng_client, enable_web):
        self.llm = self._init_llm()
        # 初始化自定义检索器
        self.retriever = HybridRetriever(
            vector_db=vector_db,
            searxng_client=searxng_client,
            enable_web=enable_web,
        )

    def _init_llm(self) -> ChatOpenAI:
        print("🔄 初始化大语言模型...")
        """初始化大语言模型"""
        return ChatOpenAI(
            openai_api_key=MODELS["llm"]["api_key"],
            model=MODELS["llm"]["model_name"],
            openai_api_base=MODELS["llm"]["api_base"],
            max_tokens=MODELS["llm"]["max_tokens"],
            streaming=True
        )

    def build_prompt(self) -> PromptTemplate:
        # 使用更健壮的模板格式
        template = """
        <|system|>
        你是一个实时信息助手，请按优先级使用以下资源：
        1. 网络搜索结果（标记为🌐）
        2. 本地知识库（标记为📚）

        {context}

        **规则**：
        1. 若网络搜索结果存在，优先使用并标注来源URL
        2. 本地知识库仅用于补充网络结果
        3. 完全无关时回答“未找到有效信息”
        </s>
        <|user|>
        问题：{question}</s>
        <|assistant|>
        """
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

    def create_chain(self) -> RetrievalQA:
        """构建问答链"""
        return RetrievalQA.from_chain_type(
            llm=self.llm.with_config(
                callbacks=[StreamingStdOutCallbackHandler()]  # 实现逐字输出效果
            ),
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.build_prompt()},
            return_source_documents=True  # 流式模式也需启用
        )

