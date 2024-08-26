from abc import ABC, abstractmethod

import torch

class Action(ABC):
    COST = 0

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
    COST = 1

    def execute(self, g):
        x = torch.zeros((1,g.x.size(1)))
        x[0, self.ntype] = 1
        g.x = torch.cat([g.x, x], dim=0)
        return g.x.size(0)-1

class DeleteNode(Action):
    COST = 1

    def execute(self, g):
        # Don't allow empty graphs
        if g.x.size(0) == 1:
            self.COST = 0
            return float('nan')

        mask = torch.ones(g.x.size(0), dtype=torch.bool)
        mask[self.target] = 0
        g.x = g.x[mask]

        mask = (g.edge_index == self.target).sum(dim=0).bool()
        g.edge_index = g.edge_index[:, ~mask]

        # Decrement index of all nodes above target
        g.edge_index[g.edge_index > self.target] -= 1

        # Cost varies bc this is technically several edge deletions
        # plus a node deletion
        self.COST = mask.sum().item() + 1

        return float('nan')

class ChangeFeature(Action):
    COST = 1

    def execute(self, g):
        feat = torch.zeros(g.x.size(1))
        feat[self.ntype] = 1
        g.x[self.target] = feat

        return self.target


class AcceptingEdges(Action):
    COST = 0

    def execute(self, g):
        g.x[self.target, -1] = 1
        return self.target

class NotAcceptingEdges(Action):
    COST = 0

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
            ]).long()
        ], dim=1)

        self.COST = accepting_edges.size(0)
        return self.target

class DeleteEdge(Action):
    '''
    Opposite of above
    '''
    def execute(self, g):
        pass