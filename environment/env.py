from copy import deepcopy
from random import choice, randint

import torch
from torch_geometric.data import Data

from environment.graph import Graph, Node
from environment.generator import generate_sample
from environment.actions import AddNode, DeleteNode, AddEdge, DeleteEdge, ChangeFeature, SIMPLE_ACTION_MAP

X = 0; EI = 1


class AStarEnv():
    def __init__(self, target_n, scrambles=10, p=0.1, n_colors=5):
        self.start = Graph(generate_sample(target_n, p), n_colors=n_colors)
        self.end = deepcopy(self.start)
        self.actual_cost = 0
        self.n_colors = n_colors

        # Mutate target graph randomly to generate initial env
        # so we know g and target are reachable within n steps
        # Target is strictly larger than initial graph
        for _ in range(scrambles):
            act = randint(0,3)

            # Add node
            if act == 0:
                target = Node(self.end.n, randint(0, n_colors-1))
                self.actual_cost += self.end.add_node(target)

            # Add edge
            elif act == 1:
                edge = (randint(0, self.end.n-1), randint(0, self.end.n-1))
                self.actual_cost += self.end.add_edge(edge)

            # Remove edge (Not very efficient...)
            elif act == 2 and len(self.end.edges):
                edges = list(self.end.edges)
                remove = choice(edges)
                self.actual_cost += self.end.remove_edge(remove)

            # Change color
            else:
                target = randint(0, self.end.n-1)
                c = randint(0, n_colors-1)
                self.actual_cost += self.end.change_color(target, c)


        self.mapping = dict()
        self.inv_mapping = dict()

    def add_remaining_nodes(self):
        '''
        After all nodes in the source graph have been mapped,
        add new nodes for any remaining nodes in dst graph that
        still need mappings
        '''
        remaining = set(range(self.end.n)) - set(self.mapping.keys())
        nid = len(self.mapping)

        cost = 0
        for dst_idx in remaining:
            # Creating node
            dst = self.end.nodes[dst_idx]
            src = Node(nid, dst.color)

            cost += self.start.add_node(src)
            cost += self.cost_function(src, dst)

        return cost

    def cost_function(self, src, dst):
        '''
        Calculates cost of mapping node from source to destination
        Adds/deletes edges if both ends are mapped, and changes node color if needed
        '''
        added_src_neighbors = set([
            n
            for n in self.start.neighbors(src.idx)
            if n in self.mapping
        ])
        added_dst_neighbors = set([
            self.inv_mapping.get(n)
            for n in self.end.neighbors(dst.idx)
            if n in self.inv_mapping
        ])

        to_add = added_dst_neighbors - added_src_neighbors
        to_remove = added_src_neighbors - added_dst_neighbors
        cost = 0

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

        return cost / self.actual_cost

    def step(self, src_idx, dst_idx):
        '''
        Select src node to map to target node.
        Assumes src input has not been assigned yet

        Returns next state, score and is_terminal
        '''
        dst = self.end.nodes[dst_idx]
        src = self.start.nodes[src_idx]

        self.mapping[src_idx] = dst_idx
        self.inv_mapping[dst_idx] = src_idx
        cost = self.cost_function(src,dst)

        # Check if finished
        # Implies any non-mapped nodes in dst graph will be node/edge insertions
        halt = False
        if len(self.mapping) == self.start.n:
            cost += self.add_remaining_nodes()
            halt = True

        return self.state(), -cost, halt

    def state(self):
        '''
        Returns combination of both graphs and a list of the source indices and
        destination indices that have yet to be mapped
        '''
        st = self.start.to_torch()
        en = self.end.to_torch()
        offset = st.x.size(0)

        src = torch.tensor([i for i in range(st.x.size(0)) if i not in self.mapping])
        dst = torch.tensor([i for i in range(en.x.size(0)) if i not in self.inv_mapping])

        '''
        # Add mappings as links between graphs
        map_src,map_dst = [],[]
        for src_n,dst_n in self.mapping.items():
            dst_n += offset
            map_src += [src_n,dst_n]
            map_dst += [dst_n,src_n]
        '''

        #mapped_ei = torch.tensor([map_src, map_dst], dtype=torch.long)

        edge_index = torch.cat([
            st.edge_index,
            en.edge_index + st.x.size(0),
            #mapped_ei
        ], dim=1)

        g = Data(
            x = torch.cat([st.x, en.x]),
            edge_index = edge_index,
            src = src,
            dst = dst + st.x.size(0),
            offset = offset,

            # Added so these can be tossed straight into RL model
            n_src = torch.tensor([src.size(0)]),
            n_dst = torch.tensor([dst.size(0)]),
            batches = torch.tensor([0, st.x.size(0) + en.x.size(0)])
        )

        return g