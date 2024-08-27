import torch
from torch_geometric.utils import to_undirected, add_remaining_self_loops, coalesce
from torch_geometric.data import Data
from torch_geometric.nn import WLConv

from environment.actions.simple_actions import Action
from environment.generator import generate_sample

class GraphEnv():
    SUCCESS_REW = 100   # Positive reward when game end reached
    PEN = -1             # For now, only penalty is number of edits made
    EXTRA_PEN = -5       # Extra punishment if nodecount/edge count is off
                        # (if graphs are trivially non-isomorphic)

    def __init__(self, n_target, n_initial=None, p=0.1, n_colors=5):
        self.nt = n_target
        self.nc = n_target if n_initial is None else n_initial
        self.p = p
        self.n_colors = n_colors
        self.wl = WLConv()

        self.reset()

    def _clean_edge_index(self, ei):
        ei = coalesce(ei)
        ei = to_undirected(ei)
        ei = add_remaining_self_loops(ei)[0]
        return ei

    def reset(self):
        target_x,target_ei = generate_sample(self.nt, self.p)
        target_ei = self._clean_edge_index(target_ei)

        current_x,current_ei = generate_sample(self.nc, self.p)
        current_ei = self._clean_edge_index(current_ei)

        self.target = Data(
            x=target_x,
            edge_index=target_ei
        )
        self.current = Data(
            x=current_x,
            edge_index=current_ei
        )

        # Need to trim off last column, as it's a feature used by
        # agent, not the node color
        self.target_hist = self.wl.histogram(self.wl(target_x[:, :-1], target_ei))
        self.ts = 0
        return self.get_state()

    def set_target(self, g: Data):
        self.target = g
        self.target_hist = self.wl.histogram(self.wl(g.x[:, :-1], g.edge_index))

    def get_state(self):
        self.current.edge_index = self._clean_edge_index(self.current.edge_index)
        return (self.current.clone(), self.target.clone())

    def step(self, a: Action):
        a.execute(self.current)
        self.current.edge_index = self._clean_edge_index(self.current.edge_index)

        # Return next_state, reward, is_terminal
        reward = self.check_finished()
        return self.get_state(), (-a.COST)+reward, reward > 0

    @torch.no_grad
    def check_finished(self):
        # Easy checks
        # Check graphs have same num nodes
        cost = 0
        if (self.target.x.size(0) != self.current.x.size(0)):
            cost += self.EXTRA_PEN
        # Check graphs have same number of nodes and node types
        if not (self.target.x[:, :-1].sum(dim=0) == self.current.x[:, :-1].sum(dim=0)).all():
            cost += self.EXTRA_PEN

        # Was trivially non-isomorphic in at least one way
        if cost != 0:
            return cost

        # Check graphs have same num edges
        if (self.target.edge_index.size(1) != self.current.edge_index.size(1)):
            return self.PEN

        # Approximate using WL Test
        hist = self.wl.histogram(self.wl(self.current.x[:, :-1], self.current.edge_index))
        if hist.size(-1) != self.target_hist.size(-1):
            return self.PEN

        return self.SUCCESS_REW if (hist == self.target_hist).all().item() else self.PEN