import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI  # 兼容DeepSeek API
from openai import OpenAI
import nltk

def init_nltk():
    """初始化NLTK资源"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("正在自动下载NLTK资源...")
        nltk.download('punkt', download_dir='./nltk_data', quiet=True)
        nltk.download('averaged_perceptron_tagger')
        nltk.data.path.append('./nltk_data')

init_nltk()
load_dotenv()

class DashScopeEmbeddings:
    """自定义 Embeddings，适配阿里云 DashScope OpenAI 兼容接口"""
    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 传入 list[str]，返回 list of embedding vectors
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        # resp.data 是一个列表，每项有 .embedding 属性
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        # 针对单条查询
        resp = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return resp.data[0].embedding


def initialize_vector_store():
    # 方案1：继续使用OpenAI Embedding（推荐）
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 方案2：使用本地Embedding模型（需下载）
    #from langchain_community.embeddings import HuggingFaceEmbeddings
    #embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    embeddings = DashScopeEmbeddings(
        model_name="text-embedding-v3",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    #loader = DirectoryLoader("docs/", glob="**/*.txt")
    # 根据后缀选择不同的 Loader
    # 同时加载 txt 和 docx
    # 2. 读取所有 docx
    # 1. 读取所有 txt
    txt_loader = DirectoryLoader("docs/", glob="**/*.txt", loader_cls= TextLoader)
    txt_docs = txt_loader.load()

    # 2. 读取所有 docx
    word_loader = DirectoryLoader(
        "docs/",
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    word_docs = word_loader.load()

    # 3. 合并
    documents = txt_docs + word_docs

    #documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="vector_store"
    )
    return vector_store


def create_qa_chain(vector_store):
    template = """根据以下上下文提供最专业的回答。如果无法回答请说明：

    上下文：{context}

    问题：{question}

    答案："""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # DeepSeek API配置
    llm = ChatOpenAI(
        model_name="deepseek-chat",
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.3
    )

    from langchain.chains import RetrievalQA
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )


def main():
    vector_store = initialize_vector_store()
    qa_chain = create_qa_chain(vector_store)

    print("DeepSeek问答系统已启动！输入'退出'结束对话")
    while True:
        question = input("\n问题：")
        if question.lower() in ["退出", "exit"]:
            break

        result = qa_chain.invoke({"query": question})
        print(f"\n答案：{result['result']}")


if __name__ == "__main__":
    main()