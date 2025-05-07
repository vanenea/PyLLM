# graph.py
from langgraph import Graph, Node

graph = Graph()
loader_node = Node(name='Load Docs', fn='loader.load_texts')
split_node = Node(name='Split Text', fn='loader.split')
vector_node = Node(name='Build VectorStore', fn='vectorstore.build')
agent_node = Node(name='Run QA Agent', fn='agent.on_message')

graph.add_nodes([loader_node, split_node, vector_node, agent_node])
# 定义流程边
graph.add_edge(loader_node, split_node)
graph.add_edge(split_node, vector_node)
graph.add_edge(vector_node, agent_node)

if __name__ == '__main__':
    graph.run()