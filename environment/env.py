from copy import deepcopy
from random import choice, randint 

from torch_geometric.data import Data 

from environment.graph import Graph, Node 
from environment.generator import generate_sample
from environment.actions import AddNode, DeleteNode, AddEdge, DeleteEdge, ChangeFeature, SIMPLE_ACTION_MAP

X = 0; EI = 1 
    

class AStarEnv(): 
    def __init__(self, target_n, scrambles=10, p=0.1, n_colors=5): 
        self.start = Graph(generate_sample(target_n, p), n_colors=n_colors)
        self.end = deepcopy(self.start)

        # Mutate target graph randomly to generate initial env 
        # so we know g and target are reachable within n steps
        # Target is strictly larger than initial graph 
        for _ in range(scrambles): 
            act = randint(0,3)

            # Add node 
            if act == 0: 
                target = Node(self.end.n, randint(0, n_colors-1))
                self.end.add_node(target)

            # Add edge 
            elif act == 1: 
                edge = (randint(0, self.end.n-1), randint(0, self.end.n-1))
                self.end.add_edge(edge)

            # Remove edge (Not very efficient...)
            elif act == 2 and len(self.end.edges): 
                edges = list(self.end.edges)
                remove = choice(edges) 
                self.end.remove_edge(remove)

            # Change color 
            else: 
                target = randint(0, self.end.n-1)
                c = randint(0, n_colors-1)
                self.end.change_color(target, c)


        self.mapping = dict()
        self.inv_mapping = dict()

    def step(self, src: Node, dst_idx: int): 
        '''
        Select src node to map to target node. 
        Assumes src input has not been assigned yet 
        
        Returns score and is_terminal 
        '''
        dst = self.end.nodes[dst_idx]
        self.mapping[src.idx] = dst_idx
        self.inv_mapping[dst_idx] = src.idx 
        cost = 0 

        # Creating node
        if src.idx >= self.start.n: 
            cost += self.start.add_node(src) 
        
        added_src_neighbors = set([
            n
            for n in self.start.neighbors(src) 
            if n in self.mapping
        ])
        added_dst_neighbors = set([
            self.inv_mapping.get(n)
            for n in self.end.neighbors(dst_idx) 
            if n in self.inv_mapping
        ])

        to_add = added_dst_neighbors - added_src_neighbors
        to_remove = added_src_neighbors - added_dst_neighbors

        # Add necessary edges 
        for n in to_add: 
            cost += self.start.add_edge((
                src.idx, n
            ))

        # Remove unnecessary edges 
        for n in to_remove: 
            cost += self.start.remove_edge((
                src.idx, n
            ))

        # Change color 
        if src.color != dst.color: 
            cost += self.start.change_color(src.idx, dst.color)

        # Check if finished 
        halt = False 
        if len(self.mapping) == self.end.n: 
            halt = True 

        return -cost, halt 