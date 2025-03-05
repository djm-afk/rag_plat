
# 创建环境，安装依赖
```bash
conda create -n rag python=3.10
pip install datasets langchain sentence_transformers tqdm chromadb langchain_wenxin
pip install openai
```

# 配置LLM的API地址
在 configs/settings.py 中修改API密钥，openai_api_base地址以及模型名称
```python
# 从环境变量读取API密钥
os.environ["LLM_API_KEY"] = "你的API密钥"  # 或通过外部注入

# 初始化兼容OpenAI的客户端
llm = ChatOpenAI(
    openai_api_key=os.getenv("LLM_API_KEY"),
    model="你的LLM模型名称",  # 对应的模型名称
    openai_api_base="你的API地址",  # API基础地址
    max_tokens=1024,
    streaming=True  # 启用流式输出
)
```

# 修改嵌入模型使用
可以在 configs/settings.py 中修改 MODEL 的值
```python
MODELS = {
    "embedding": {
        "device": "cpu",  # 可换成gpu cuda加速
    }
}
```

# 运行main.py
```python
python main.py
```
