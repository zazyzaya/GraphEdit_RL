import torch
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.data import Data
from torch_geometric.nn import WLConv

from actions.simple_actions import Action

class GraphEnv():
    SUCCESS_REW = 100   # Positive reward when game end reached
    EXTRA_PEN = 2       # Extra punishment if nodecount/edge count is off
                        # (if graphs are trivially non-isomorphic)

    def __init__(self, n, p=0.1, n_colors=5):
        self.n = n
        self.p = p
        self.n_colors = n_colors
        self.wl = WLConv()

        self.reset()

    def reset(self):
        target = erdos_renyi_graph(self.n, self.p)
        target_x = torch.zeros(self.n, self.n_colors)
        target_x[:, torch.randint(high=self.n_colors, size=self.n)] = 1

        # May want to start agent with single node?
        current = erdos_renyi_graph(self.n, self.p)
        current_x = torch.zeros(self.n, self.n_colors)
        current_x[:, torch.randint(high=self.n_colors, size=self.n)] = 1

        self.target = Data(
            x=target_x,
            edge_index=target
        )
        self.current = Data(
            x=current_x,
            edge_index=current
        )

        self.target_hist = self.wl(target_x, target)
        self.ts = 0

    def step(self, a: Action):
        a.execute(self.current)

        # Return next_state, reward, is_terminal
        reward = self.check_finished()
        return self.current.clone(), reward, reward >= 0

    def check_finished(self):
        # Easy checks
        # Check graphs have same num nodes
        if (self.target.x.size(0) != self.current.x.size(0)):
            return -1*self.EXTRA_PEN
        # Check graphs have same num edges
        if (self.target.edge_index.size(1) != self.current.edge_index.size(1)):
            return -1*self.EXTRA_PEN
        # Check graphs have same number of nodes and node types
        if not (self.target.x.sum(dim=0) == self.current.x.sum(dim=0)).all():
            return -1*self.EXTRA_PEN

        # Approximate using WL Test
        hist = self.wl.histogram(self.wl(self.current.x, self.current.edge_index))
        return self.SUCCESS_REW if (hist == self.target_hist).all().item() else -1