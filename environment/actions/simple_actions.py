from abc import ABC, abstractmethod

import torch

class Action(ABC):
    def __init__(self, target=None, ntype=None):
        self.target = target
        self.ntype = ntype

    @abstractmethod
    def execute(self, g):
        '''
        Return target node's new ID
        Or float('nan') if it was deleted
        '''
        pass

class AddNode(Action):
    def execute(self, g):
        x = torch.zeros((1,g.x.size(1)))
        x[0, self.ntype] = 1
        g.x = torch.cat([g.x, x], dim=0)
        return g.x.size(0)-1

class DeleteNode(Action):
    def execute(self, g):
        mask = torch.ones(g.x.size(0), dtype=torch.bool)
        mask[self.target] = 0
        g.x = g.x[mask]

        mask = (g.edge_index == self.target).sum(dim=0)
        g.edge_index = g.edge_index[:, mask]
        return float('nan')

class ChangeFeature(Action):
    def execute(self, g):
        feat = torch.zeros(g.x.size(1))
        feat[self.ntype] = 1
        g.x[self.target] = feat

        return self.target


class AcceptingEdges(Action):
    def execute(self, g):
        g.x[self.target, -1] = 1
        return self.target

class NotAcceptingEdges(Action):
    def execute(self, g):
        g.x[self.target, -1] = 0
        return self.target

class AddEdge(Action):
    '''
    Adds an edge from the target node to all nodes that are
    accepting edges (extra feature)
    This was the best way I could think of to make edge additions
    a node-level action. Requires multi-step planning to select target
    node, then to add edges from another node
    '''
    def execute(self, g):
        accepting_edges = (g.x[:, -1] == 1).nonzero().flatten()

        g.edge_index = torch.cat([
            g.edge_index,
            torch.tensor([
                [self.target] * accepting_edges.size(0),
                accepting_edges.tolist()
            ])
        ], dim=1)

        return self.target