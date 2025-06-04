from loguru import logger
import numpy as np
from .graph import MutateGraph, MutateNode
from dataclasses import dataclass

@dataclass
class SelectionResult:
    """
    A class to represent the result of a selection process.
    """
    # The mutated query
    query: str      = ""
    # The reward received for the query
    rewards: float  = 0.0
    or_score: float = 0.0
    toxic_rate:float = 0.0
    answer_rate: float = 0.0
    diversity_score: float = 0.0
    # The mutation applied to the query
    mutation: str   = ""
    # If we enable the mutator update circle, this will be the updated prompt
    mutation_prompt: str = ""

class BaseSelection:
    def __init__(self):
        pass
    
    def select(self, num:int) -> list[str]:
        """
        Selects a number of queries from the pool.
        :param num: The number of questions to select.
        :return: A list of selected questions.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def update(self, results: dict[str, list[SelectionResult]], new_seeds: dict[str, list[SelectionResult]]) -> None:
        """
        Updates the selection pool with the results of the previous generation.
        :param results: A dictionary containing the results of the previous generation.
        :param new_seeds: A dictionary containing the new seeds.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class RandomSelection(BaseSelection):
    def __init__(self, pool: list[str]):
        super().__init__()
        self.pool = pool
        self.selected = []
    
    def select(self, num: int) -> list[str]:
        """
        Selects a number of queries randomly from the pool.
        :param num: The number of questions to select.
        :return: A list of selected questions.
        """
        import random
        self.selected = random.sample(self.pool, num)
        return self.selected
    def update(self, results: dict[str, list[SelectionResult]], new_seeds: dict[str, list[SelectionResult]]) -> None:
        """
        Updates the selection pool with the results of the previous generation.
        :param results: A dictionary containing the results of the previous generation.
        :param new_seeds: A dictionary containing the new seeds.
        """
        for v in new_seeds.values():
            for q in v:
                if q.query not in self.pool:
                    self.pool.append(q.query)

class RoundRobinSelection(BaseSelection):
    def __init__(self, pool: list[str]):
        super().__init__()
        self.pool = pool
        self.index = len(pool) - 1
    
    def select(self, num: int) -> list[str]:
        """
        Selects a number of queries in a round-robin fashion from the pool.
        :param num: The number of questions to select.
        :return: A list of selected questions.
        """
        selected = []
        for _ in range(num):
            selected.append(self.pool[self.index])
            self.index = (self.index - 1) % len(self.pool)
        return selected
    
    def update(self, results: dict[str, list[SelectionResult]], new_seeds: dict[str, list[SelectionResult]]) -> None:
        """
        Updates the selection pool with the results of the previous generation.
        :param results: A dictionary containing the results of the previous generation.
        :param new_seeds: A dictionary containing the new seeds.
        """
        for v in new_seeds.values():
            for q in v:
                if q.query not in self.pool:
                    self.pool.append(q.query)
                    
class UCBSelection(BaseSelection):
    def __init__(self, pool: list[str], explore_rate: float = 0.2):
        super().__init__()
        self.pool = pool
        self.reward = np.zeros(len(pool), dtype=float)
        self.selected_num = np.zeros(len(pool), dtype=int)
        self.step = 1
        self.explore_rate = explore_rate
        self.last_selected:list[int] = []
    
    def select(self, num: int) -> list[str]:
        """
        Selects a number of queries in a UCB fashion from the pool.
        :param num: The number of questions to select.
        :return: A list of selected questions.
        """
        
        smooth_selected_num = self.selected_num.astype(float) + 1
        scores = self.reward / smooth_selected_num + self.explore_rate * np.sqrt(2 * np.log(self.step) / smooth_selected_num)
        self.last_selected = np.argsort(scores)[-num:]
        return [self.pool[i] for i in self.last_selected]
    
    def update(self, results: dict[str, list[SelectionResult]], new_seeds: dict[str, list[SelectionResult]]) -> None:
        """
        Updates the selection pool with the results of the previous generation.
        :param results: A dictionary containing the results of the previous generation.
        :param new_seeds: A dictionary containing the new seeds.
        """
        self.step += 1
        result_list = []
        for v in new_seeds.values():
            for q in v:
                result_list.append(q.query)
                if q.query not in self.pool:
                    self.pool.append(q.query)
                    self.reward = np.append(self.reward, 0)
                    self.selected_num = np.append(self.selected_num, 0)
        rewards = {k: np.mean([v.rewards for v in r]) for k, r in results.items()}
        for selected in self.last_selected:
            if self.pool[selected] in rewards:
                self.reward[selected] += rewards[self.pool[selected]]
                self.selected_num[selected] += 1
        self.last_selected = []
        
class MCTSExploreSelection(BaseSelection):
    def __init__(self, pool: list[str]|MutateGraph, explore_rate:float=0.5, alpha:float=0.1, beta:float=0.2):
        super().__init__()
        if isinstance(pool, list):
            self.graph = MutateGraph()
            for i, q in enumerate(pool):
                self.graph.add_node(q, "query", "root")
        elif isinstance(pool, MutateGraph):
            self.graph = pool
        else:
            raise ValueError("pool should be a list of queries or a MutateGraph instance.")
        self.step = 1
        self.explore_rate = explore_rate
        self.alpha = alpha
        self.beta = beta
        self.last_selected:list[MutateNode] = []
        self.select_path:list[list[MutateNode]] = []
        
    def select(self, num: int) -> list[str]:
        """
        Selects a number of queries in a MCTS fashion from the pool.
        :param num: The number of questions to select.
        :return: A list of selected questions.
        """
        scores = np.array([pn.reward / (pn.selected_num + 1) + self.explore_rate * np.sqrt(2 * np.log(self.step) / (pn.selected_num + 0.01)) for pn in self.graph.root.children])
        init_nodes = np.argsort(scores)[-num:]
        init_nodes:list[MutateNode] = [self.graph.root.children[i] for i in init_nodes]
        self.select_path = []
        self.last_selected = []
        
        for i, node in enumerate(init_nodes):
            select_path = [node]
            cur = node
            while len(cur.children) > 0:
                if cur.node_type == "query" and np.random.rand() < self.alpha:
                    break
                # we follow the settings of GPTFuzzer, where the selected number adds 0.01 here
                cur = max(
                    cur.children,
                    key=lambda pn:
                        pn.reward / (pn.selected_num + 1) + self.explore_rate * np.sqrt(2 * np.log(self.step) / (pn.selected_num + 0.01))
                )
                select_path.append(cur)
            self.select_path.append(select_path)
            self.last_selected.append(cur)
        return [node.node_id for node in self.last_selected]
    def update(self, results: dict[str, list[SelectionResult]], new_seeds: dict[str, list[SelectionResult]]) -> None:
        """
        Updates the selection pool with the results of the previous generation.
        :param results: A dictionary containing the results of the previous generation.
        """
        self.step += 1
        result_list = []
        for k, v in new_seeds.items():
            for q in v:
                result_list.append(q.query)
                if q.query not in self.graph.nodes:
                    self.graph.add_node(q.query, "query", k, mutation=q.mutation)
        rewards = {k: np.mean([v.rewards for v in r]) for k, r in results.items()}
        for path in self.select_path:
            for node in path:
                node.reward += rewards[path[-1].node_id]*max(self.beta, (1-0.1*path[-1].level))
                node.selected_num += 1

        self.last_selected = []
        self.select_path = []
        
        # logger.info(f"graph: {self.graph}")
        for node in self.graph.root.children:
            if node.node_type == "classification":
                logger.info(f"node: {node.node_id}, reward: {node.reward}, selected_num: {node.selected_num}")
        
def get_selector(selection_type: str, pool: list[str]|MutateGraph, *args, **kwargs) -> BaseSelection:
    """
    Returns a selection object based on the selection type.
    :param selection_type: The type of selection to use.
    :param pool: The pool of queries to select from.
    :return: A selection object.
    """
    if selection_type.lower() == "random":
        return RandomSelection(pool)
    elif selection_type.lower() == "round_robin":
        return RoundRobinSelection(pool)
    elif selection_type.lower() == "ucb":
        return UCBSelection(pool, *args, **kwargs)
    elif selection_type.lower() == "mcts_explore":
        return MCTSExploreSelection(pool, *args, **kwargs)
    else:
        raise ValueError(f"Unknown selection type: {selection_type}")