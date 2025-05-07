# main.py
import os
from dotenv import load_dotenv
from loader import DocsLoader
from vectorstore import VectorStore
from agent import QAAgent

load_dotenv()
key = os.getenv('OPENAI_API_KEY')

# 1. 加载并切分文档
loader = DocsLoader('docs/')
docs = loader.load_texts()
chunks = loader.split(docs)

# 2. 建立向量数据库
vs = VectorStore(api_key=key)
vs.build(chunks)

# 3. 启动控制台循环
agent = QAAgent(vs)
print("欢迎使用 Docs QA 工具，输入问题或输入 exit 退出。")
while True:
    q = input("\n问题： ")
    if q.lower() in ('exit', 'quit'):
        break
    answer = agent.on_message(q)
    print("\n回答：", answer)
print("已退出。")