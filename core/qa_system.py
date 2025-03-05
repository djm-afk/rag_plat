from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai.chat_models import ChatOpenAI
from configs.settings import MODELS


class QASystemBuilder:
    """问答系统构造器"""

    def __init__(self, vector_db):
        self.retriever = self._init_retriever(vector_db=vector_db)
        self.llm = self._init_llm()

    def _init_retriever(self, vector_db):
        return vector_db.as_retriever(
            search_type="similarity_score_threshold",  # 默认相似度搜索
            search_kwargs={
                "k": 3,                 # 召回数量
                "score_threshold": 0.6  # 相似度门槛
            },
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
        """构造提示词模板"""
        template = """
        <|system|>
        你是一个农业知识专家，请严格根据以下上下文回答问题，上下文中没有的就回答“上下文中未找到相关信息”：
        {context}
        </s>
        <|user|>
        问题：{question}</s>
        <|assistant|>
        """
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
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

