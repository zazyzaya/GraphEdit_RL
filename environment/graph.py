from typing import Union 

import torch 
from torch_geometric.data import Data 

class Node(): 
    def __init__(self, idx: int, color: int): 
        self.idx = idx 
        self.color = color 

    def __str__(self): 
        return f'<nid: {self.idx}, color: {self.color}>'
    
    def __repr__(self):
        return self.__str__()

class Graph(): 
    def __init__(self, data: Data, n_colors: int, 
                 add_n_cost=1, add_e_cost=1, rem_e_cost=1, color_cost=1): 
        self.n_colors = n_colors 
        self.add_n_cost = add_n_cost
        self.add_e_cost = add_e_cost
        self.rem_e_cost = rem_e_cost
        self.color_cost = color_cost

        self.nodes = [
            Node(i, data.x[i].item())
            for i in range(data.x.size(0))
        ]
        self.edges = set([
            (data.edge_index[0,i].item(), data.edge_index[1,i].item())
            for i in range(data.edge_index.size(1))
        ])
        self.n = len(self.nodes)

    def add_edge(self, edge): 
        if edge[0] !=  edge[1]: 
            self.edges.add(edge)
            reverse = (edge[1], edge[0])
            self.edges.add(reverse)

            return self.add_e_cost
        return 0 

    def remove_edge(self, edge): 
        if edge[0] != edge[1] and edge in self.edges:
            self.edges.discard(edge) 
            reverse = (edge[1], edge[0])
            self.edges.discard(reverse) 

            return self.rem_e_cost
        return 0

    def add_node(self, node: Node): 
        if node.color > self.n_colors: 
            raise ValueError(f"Nodes must have color <{self.n_colors}")
        
        self.n += 1 
        self.nodes.append(node)

        return self.add_n_cost

    def change_color(self, node_idx: int, color: int): 
        if color > self.n_colors: 
            raise ValueError(f"Nodes must have color <{self.n_colors}")
        
        self.nodes[node_idx].color = color 
        return self.color_cost

    def neighbors(self, node: Union[int, Node]):
        '''
        TODO make more efficient by adding CSR repr or something 
        '''
        if isinstance(node, Node): 
            node = node.idx 

        neigh = []
        for edge in self.edges: 
            if edge[0] == node: 
                neigh.append(edge[1])

        return neigh 

    def to_torch(self): 
        n = len(self.nodes)
        x = torch.zeros(n, self.n_colors)
        x[torch.arange(n), [node.color for node in self.nodes]]

        src,dst = [],[]
        for (s,d) in self.edges:
            src.append(s)
            dst.append(d) 

        ei = torch.tensor([src,dst])
        return Data(
            x=x, edge_index=ei
        )
    