from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# 路径配置
PATHS = {
    "source_file": PROJECT_ROOT / "data" / "藜.md",
    "vector_db": PROJECT_ROOT / "storage" / "chroma_db"
}

# 模型配置
MODELS = {
    "embedding": {
        "name": "moka-ai/m3e-base",
        "device": "cpu",  # 可换成gpu cuda加速
        "encode_kwargs": {
            "normalize_embeddings": True,
            "batch_size": 32
        }
    },
    "llm": {
        "api_key": "gpustack_a9d1a18b4b60be87_d13dd2ccc57ca957ac1590728f3d3479",
        "model_name": "deepseek-r1-1.5B",
        "api_base": "http://192.168.18.93:4080/v1-openai",
        "max_tokens": 1024
    }
}

# 处理参数
PROCESSING = {
    "encodings": ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"],
    "markdown_headers": [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3")
    ]
}