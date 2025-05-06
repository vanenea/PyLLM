import os

# 文档读取
import docx2txt

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# LangGraph（可选，用于图谱构建）
from langgraph import Graph, Node

# 初始化 Embeddings & Vector Store
EMBEDDING_MODEL = "openai-embedding"
embeddings = OpenAIEmbeddings()
vector_store = None

graph = Graph()

def process_docx_file(file_path: str):
    """
    读取并处理指定路径的 DOCX 文档，生成向量索引和知识图谱节点。
    """
    text = docx2txt.process(file_path)
    # 文本拆分
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_text(text)

    # 向量化并存储
    global vector_store
    vector_store = Chroma.from_texts(docs, embeddings)

    # 构建知识图谱节点
    for i, chunk in enumerate(docs):
        node = Node(id=f"chunk_{i}", text=chunk)
        graph.add_node(node)


def interactive_qa():
    """
    控制台交互式问答循环，使用深度检索模型（deepseek）。
    """
    # 构建检索式问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="deepseek"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    print("文档已加载，输入问题进行问答，输入 'exit' 或 'quit' 结束。")
    while True:
        question = input("您: ")
        if question.lower() in {"exit", "quit"}:
            print("退出程序。")
            break
        answer = qa_chain.run(question)
        print(f"回答: {answer}\n")


if __name__ == "__main__":
    DOC_PATH = "/opt/1.docx"
    if not os.path.exists(DOC_PATH):
        raise FileNotFoundError(f"文档未找到：{DOC_PATH}")

    print(f"正在加载文档：{DOC_PATH}")
    process_docx_file(DOC_PATH)
    interactive_qa()