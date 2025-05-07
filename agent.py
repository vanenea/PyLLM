# agent.py
from SmolAgents import SmolAgent
from vectorstore import VectorStore

class QAAgent(SmolAgent):
    def __init__(self, vs: VectorStore):
        super().__init__()
        self.vs = vs

    def on_message(self, message: str):
        # 1. 从向量库检索相关 chunks
        results = self.vs.query(message)
        # 2. 拼接上下文
        context = "\n\n".join([r[0].page_content for r in results])
        # 3. 调用 LLM 生成回答
        resp = self.call_llm(f"根据以下内容回答：\n{context}\n问题：{message}")
        return resp