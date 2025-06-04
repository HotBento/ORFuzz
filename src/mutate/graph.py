from typing import Optional

class MutateNode:
    def __init__(self, node_id: str, node_type: str, parent:Optional[str]=None, mutation:Optional[str]=None, level:int=0):
        """
        Initialize a MutateNode instance.
        :param node_id: The unique identifier for the node.
        :param node_type: The type of the node (e.g., "root", "query").
        :param parent: The parent node of this node (if any).
        :param mutation: The mutation associated with this node (if any).
        """
        self.node_id:str = node_id
        self.node_type:str = node_type
        self.children:list[MutateNode] = []
        self.mutation:Optional[str] = mutation
        self.parent:Optional[str] = parent
        # if parent == None:
        #     self.level:int = 0
        # else:
        #     self.level:int = parent.level + 1
        self.level:int = level
        self.reward:float = 0.0
        self.selected_num:int = 0

    def __repr__(self):
        return f"MutateNode(node_id={self.node_id}, node_type={self.node_type}, reward={self.reward}, selected_num={self.selected_num}, mutation={self.mutation})"

class MutateGraph:
    def __init__(self):
        """
        Initialize a MutateGraph instance.
        """
        self.nodes:dict[str, MutateNode] = dict()
        self.root = MutateNode("root", "root")
        self.nodes["root"] = self.root
        
    def __repr__(self):
        return f"MutateGraph(nodes={self.nodes})"

    def add_node(self, node_id: str, node_type: str, parent_id: str, mutation=None):
        """
        Add a node to the graph.
        :param node_id: The unique identifier for the node.
        :param node_type: The type of the node (e.g., "function", "variable").
        :param parent_id: The unique identifier for the parent node (if any).
        :param mutation: The mutation associated with this node (if any).
        """
        new_node = MutateNode(node_id, node_type, parent=parent_id, mutation=mutation, level=self.nodes[parent_id].level + 1)
        self.nodes[node_id] = new_node
        parent_node = self.nodes[parent_id]
        new_node.parent = parent_node
        parent_node.children.append(new_node)
        return new_node
    def get_node(self, node_id: str):
        """
        Get a node from the graph by its ID.
        :param node_id: The unique identifier for the node.
        :return: The node with the specified ID, or None if it doesn't exist.
        """
        return self.nodes.get(node_id, None)
    def get_children(self, node_id: str):
        """
        Get the children of a node.
        :param node_id: The unique identifier for the node.
        :return: A list of child nodes.
        """
        node = self.get_node(node_id)
        if node:
            return node.children
        return []
    def get_parent(self, node_id: str):
        """
        Get the parent of a node.
        :param node_id: The unique identifier for the node.
        :return: The parent node, or None if it doesn't exist.
        """
        node = self.get_node(node_id)
        if node:
            return node.parent
        return None
    def get_all_nodes(self):
        """
        Get all nodes in the graph.
        :return: A list of all nodes.
        """
        return list(self.nodes.values())