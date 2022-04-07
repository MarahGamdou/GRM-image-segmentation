import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class FordFulkerson():
    def __init__(self, graph):
        self.graph = graph 
        # self.n_nodes = n_nodes
        # self.graph = nx.DiGraph()
        # n = int(np.sqrt(n_nodes))
        # self.graph.add_node(0, pos=(0, n + 1))
        # self.graph.add_node(-1, pos=(n + 1, n + 1))
        # for num_node in range(1, n_nodes + 1):
        #     self.graph.add_node(num_node, pos=((num_node - 1) // n, (num_node - 1) % n))
        # self.add_egdes(edges)
        self.frozen_graph = self.graph.copy()

    def add_egdes(self, edges):
        """Add edges to the graph

        Args:
            edges (list): edges with their capacities 
        """
        for edge in edges:
            i, j, c = edge
            self.graph.add_edge(i, j, capacity=c, flow=0)

    def find_path(self, s, t):
        """Find all possible paths between s and t

        Args:
            s (int): source
            t (int): sink

        Returns:
            generator : all paths between s and t 
        """
        paths = nx.all_simple_paths(self.dummy_graph, source=s, target=t)
        return paths

    def get_possible_flow(self, path):
        """Get maximum flow allowable for a given path

        Args:
            path (list): a given path

        Returns:
           int: allowable flow 
        """
        flow = []
        for i in range(1, len(path)):
            edge_data = self.graph.get_edge_data(path[i - 1], path[i])
            flow.append(edge_data["capacity"] - edge_data["flow"])
        return np.min(flow)

    def select_min_path(self, paths):
        """Select the minimum s-t path

        Args:
            paths (iterator): all paths 

        Returns:
            iterator, int :mininmum path, respective flow 
        """
        for path in paths:
            flow = self.get_possible_flow(path)
            if flow > 0:
                return path, flow
        return None, None

    def min_cut(self, s, t):
        """find min-cut 

        Args:
            s (int): source
            t (int): sink
        """
        self.dummy_graph = self.graph.copy()
        paths = self.find_path(s, t)
        while (nx.has_path(self.dummy_graph, s, t)) > 0:
            path, possible_flow = self.select_min_path(paths)
            if path:
                for i in range(1, len(path)):
                    u = path[i - 1]
                    v = path[i]
                    if not(self.graph.has_edge(v, u)):
                        self.graph.add_edge(v, u, capacity=0, flow=0)
                    current_flow = self.graph.get_edge_data(u, v)["flow"]
                    reverse_capacity = self.graph.get_edge_data(v, u)["capacity"]
                    self.graph.update([(u, v,
                                        {"flow": current_flow + possible_flow,
                                         "capacity": self.graph.get_edge_data(u, v)["capacity"]}),
                                       (v, u,
                                        {"capacity": reverse_capacity + possible_flow,
                                         "flow": self.graph.get_edge_data(v, u)["flow"] or 0})])
                    if self.graph.get_edge_data(u, v)["capacity"] == self.graph.get_edge_data(u, v)["flow"]:
                        self.dummy_graph.remove_edge(u, v)
            paths = self.find_path(s, t)
        for edge in self.frozen_graph.edges():
            u, v = edge
            self.frozen_graph.update([(u,
                                       v,
                                       {"capacity": self.frozen_graph.get_edge_data(u, v)["capacity"],
                                        "flow": self.graph.get_edge_data(u, v)["flow"]})])