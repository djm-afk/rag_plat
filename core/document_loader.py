from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document
from configs.settings import PATHS, PROCESSING


class DocumentLoader:
    """支持多种编码的文档加载器"""

    def __init__(self):
        self.file_path = PATHS["source_file"]
        self.encodings = PROCESSING["encodings"]

    def load_with_fallback(self) -> List[Document]:
        print("🔄 加载文件中...")
        """自动尝试不同编码加载文档"""
        for encoding in self.encodings:
            try:
                loader = TextLoader(self.file_path, encoding=encoding)
                return loader.load()
            except (UnicodeDecodeError, RuntimeError):
                continue
        raise RuntimeError(f"无法加载文件 {self.file_path}")


class MarkdownProcessor:
    """Markdown文档处理器"""

    def __init__(self):
        self.headers = PROCESSING["markdown_headers"]

    def split_document(self, content: str) -> List[Document]:
        print("🔄 执行多级分块...")
        """执行多级分块"""
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers,
            return_each_line=False,
            strip_headers=False
        )
        return splitter.split_text(content)