import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class PushRelabel():
    def __init__(self, graph, source, sink):
        self.graph = graph 
        self.source = source 
        self.sink = sink 
        self.V = len(self.graph.nodes())
        height = 0
        nx.set_node_attributes(self.graph, height, "height")
        # height of the source is equal to the number of all vertices
        self.graph.update(nodes=[(self.source,{'height':self.V})])
        
        excess = 0
        nx.set_node_attributes(self.graph, excess, "excess")

        for n in self.graph.neighbors(self.source):
            # flow of edges from source is equal to their capacities
            self.graph[self.source][n]['flow'] = self.graph[self.source][n]['capacity']
            self.graph.update(nodes=[(n,{'excess':self.graph[self.source][n]['capacity']})])
            lexcess = nx.get_node_attributes(self.graph, 'excess')
            self.graph.update(nodes=[(self.source,{'excess': lexcess[self.source] - self.graph[self.source][n]['capacity']})])
            # residual graph 
            if not(self.graph.has_edge(n, self.source)):
                self.graph.add_edge(n, self.source, capacity=0, flow=0)
            self.graph[n][self.source]['flow']= - self.graph[self.source][n]['capacity']
            
        
    def excess_node(self):
        """Find a node with excess flow 

        Returns:
            int: node number 
        """
        lexcess = nx.get_node_attributes(self.graph, 'excess')
        for v in range(self.V):
            if v != self.source and v != self.sink and lexcess[v] > 0:
                return v
        return None
    
    def push(self, node):
        """Send flow to neighbors

        Args:
            node (int): node with excess flow 

        Returns:
            boolean: if flow was sent to neighbors 
        """
        lexcess = nx.get_node_attributes(self.graph, 'excess')
        assert(lexcess[node]> 0)
        for v in self.graph.neighbors(node):
            lheights = nx.get_node_attributes(self.graph, 'height')
            lexcess = nx.get_node_attributes(self.graph, 'excess')
            
            if (self.graph[node][v]['capacity'] > self.graph[node][v]['flow']) and (lheights[node] == lheights[v] + 1):  #teste avec > 
                flow = min(self.graph[node][v]['capacity'] - self.graph[node][v]['flow'], lexcess[node])
                
                self.graph[node][v]['flow'] += flow

                if node not in self.graph[v].keys():
                    self.graph.add_edge(v, node, capacity=0, flow=0)
                if self.graph[v][node]['capacity'] > self.graph[v][node]['flow']:
                    self.graph[v][node]['flow'] -= flow
                else:
                    self.graph[v][node]['flow'] = 0
                    self.graph[v][node]['capacity'] = flow
                    
                self.graph.update(nodes=[(node,{'excess':lexcess[node]-flow})])
                self.graph.update(nodes=[(v,{'excess':lexcess[v]+flow})])
                return True
        return False
        
    def relabel(self, node):
        """Increase the height of a node 

        Args:
            node (int): node whom the height will be increased 
        """
        lheights = nx.get_node_attributes(self.graph, 'height')
        assert([lheights[node] <= lheights[v] for v in self.graph.neighbors(node) if self.graph[node][v]['capacity'] > self.graph[node][v]['flow']])
        self.graph.update(nodes=[(node,
                                  {'height':  1 + min([lheights[v] for v in self.graph.neighbors(node) if self.graph[node][v]['capacity'] > self.graph[node][v]['flow']])})])
        
    
    def min_cut(self):
        """Find min cut and update graph 
        """
        while True:
            node = self.excess_node()
            if node == None: break
            if not self.push(node):
                self.relabel(node)
        lexcess = nx.get_node_attributes(self.graph, 'excess')
        print("Max flow", lexcess[self.sink])
    
        capacities = nx.get_edge_attributes(self.graph, 'capacity')
        # flows = nx.get_edge_attributes(self.graph, 'flow')
        # cuts = []
        # for edge in capacities.keys():
        #     if (capacities[edge]==flows[edge]) and (capacities[edge]>0) : 
        #         cuts.append(edge)
        
        for edge in capacities.keys():
            if (capacities[edge]<=0):
                self.graph.remove_edge(edge[0], edge[1])
                
        # print('cuts')
        # print(cuts)
        
        

       
        
        
        
        

    
    
        