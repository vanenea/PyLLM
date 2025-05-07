from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocsLoader:
    def __init__(self, folder: str):
        self.folder = folder

    def load_texts(self):
        texts = []
        for file in os.listdir(self.folder):
            path = os.path.join(self.folder, file)
            parts = partition(path)
            raw = "\n".join([p.get_text() for p in parts])
            texts.append({'filename': file, 'content': raw})
        return texts

    def split(self, texts, chunk_size=1000, overlap=200):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        chunks = []
        for doc in texts:
            for chunk in splitter.split_text(doc['content']):
                chunks.append({'text': chunk, 'source': doc['filename']})
        return chunks