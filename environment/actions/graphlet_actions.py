import torch

from environment.actions.simple_actions import Action

def update(g, new_x, new_edges):
    g.x = torch.cat([g.x, new_x], dim=0)
    g.edge_index = torch.cat([g.edge_index, new_edges], dim=1)

class G1(Action):
    # (X) -- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((1, g.x.size(1)))
        new_x[0, self.ntype] = 1

        new_edge = torch.tensor([[self.target], [nid]])
        update(g, new_x, new_edge)
        return self.target

class G2(Action):
    # (X) -- ( )
    #  |
    # ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((2, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([[self.target]*2, [nid,nid+1]])
        update(g, new_x, new_edges)
        return self.target

class G3(Action):
    # (X) -- ( )
    #  | \
    # ( ) ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((3, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([[self.target]*3, [nid,nid+1,nid+2]])
        update(g, new_x, new_edges)
        return self.target

class G4(Action):
    # ( ) --(X) -- ( )
    #       / \
    #     ( ) ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([[self.target]*4, [nid,nid+1,nid+2,nid+3]])
        update(g, new_x, new_edges)
        return self.target

class G5(Action):
    # (X) -- ( )
    #  |
    # ( ) -- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((3, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([[self.target,self.target,nid], [nid,nid+1,nid+2]])
        update(g, new_x, new_edges)
        return self.target

class G6(Action):
    #    (X)
    #   /    \
    # ( ) -- ( )
    def execute(self, g,):
        nid = g.x.size(0)
        new_x = torch.zeros((2, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,self.target,nid],
            [nid,nid+1,nid+1]]
        )
        update(g, new_x, new_edges)
        return self.target

class G7(Action):
    #    (X) -- ( )
    #   /   \
    # ( ) -- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((3, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,self.target,self.target,nid],
            [nid,nid+1,nid+2,nid+1]
        ])
        update(g, new_x, new_edges)
        return self.target

class G8(Action):
    # ( ) --(X) -- ( )
    #      /   \
    #    ( ) -- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([[self.target]*4 + [nid], [nid,nid+1,nid+2,nid+3,nid+1]])
        update(g, new_x, new_edges)
        return self.target


class G9(Action):
    #    (X) -- ( )
    #   /      /
    # ( ) -- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((3, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,nid,nid+1,nid+2],
            [nid,nid+1,nid+2,self.target]
        ])
        update(g, new_x, new_edges)
        return self.target


class G10(Action):
    #( ) -- (X) -- ( )
    #      /      /
    #    ( ) -- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,nid,nid+1,nid+2, self.target],
            [nid,nid+1,nid+2,self.target, nid+3]
        ])
        update(g, new_x, new_edges)
        return self.target


class G11(Action):
    #    (X) -- ( )
    #   /   \  /
    # ( ) -- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((3, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,nid,nid+1,nid+2,self.target,],
            [nid,nid+1,nid+2,self.target,nid+1]
        ])
        update(g, new_x, new_edges)
        return self.target


class G12(Action):
    # ( ) --(X)    ( )
    #      /   \  /
    #    ( ) -- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target]*3 + [nid+1,nid+2],
            [nid,nid+1,nid+2, nid+2,nid+3]])
        update(g, new_x, new_edges)
        return self.target

class G13(Action):
    # ( ) --(X) -- ( )
    #      /   \  /
    #    ( ) -- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target]*4     +  [nid+1,nid+2],
            [nid,nid+1,nid+2,nid+3, nid+2,nid+3]])
        update(g, new_x, new_edges)
        return self.target

class G14(Action):
    # ( ) -- (X) -- ( )
    #  |     /      /
    #  |  ( ) -- ( )
    #  \----------|
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target]*3 + [nid+2]*3,
            [nid,nid+1,nid+3,  nid,nid+1,nid+3]])
        update(g, new_x, new_edges)
        return self.target


class G15(Action):
    # ( ) -- (X) -- ( )
    #  |     /  \   /
    #  |  ( ) -- ( )
    #  \----------|
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target]*4    +   [nid+2]*3,
            [nid,nid+1,nid+2,nid+3, nid,nid+1,nid+3]])
        update(g, new_x, new_edges)
        return self.target

class G16(Action):
    # ( )   (X) -- ( )
    #   \   /     /
    #    ( )    ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [nid,nid+1, self.target,nid+2],
            [nid+1,self.target, nid+2,nid+3]
        ])
        update(g, new_x, new_edges)
        return self.target

class G17(Action):
    # ( ) -- (X) -- ( )
    #   \   /       /
    #    ( )      ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [nid,nid+1,self.target,  self.target,nid+2],
            [nid+1,self.target,nid,  nid+2,nid+3]
        ])
        update(g, new_x, new_edges)
        return self.target


class G18(Action):
    # ( ) -- (X) -- ( )
    #   \   /   \   /
    #    ( )     ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target]*4 +      [nid,nid+2],
            [nid,nid+1,nid+2,nid+3, nid+1,nid+3]
        ])
        update(g, new_x, new_edges)
        return self.target


class G19(Action):
    # ( ) -- (X) -- ( )
    #   \           /
    #    ( ) --- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,nid,nid+1,nid+2,nid+3],
            [nid,      nid+1,nid+2,nid+3,self.target]
        ])
        update(g, new_x, new_edges)
        return self.target


class G20(Action):
    # ( ) -- (X) -- ( )
    #   \   /       /
    #    ( ) --- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,nid,nid+1,nid+2,nid+3,  self.target],
            [nid,      nid+1,nid+2,nid+3,self.target,  nid+1]
        ])
        update(g, new_x, new_edges)
        return self.target

class G21(Action):
    # ( ) -- (X) -- ( )
    #   \   /   \   /
    #    ( ) --- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,nid,nid+1,nid+2,nid+3,  self.target,self.target],
            [nid,      nid+1,nid+2,nid+3,self.target,  nid+1,nid+2]
        ])
        update(g, new_x, new_edges)
        return self.target


class G22(Action):
    # ( ) -- (X) -- ( )
    #  |\   /
    #  | ( )
    #  |   \
    #  \-- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target]*3 +  [nid,  nid+1,nid+2],
            [nid,nid+1,nid+3,   nid+1,nid+2,nid]
        ])
        update(g, new_x, new_edges)
        return self.target

class G23(Action):
    # ( ) -- (X)
    #  |\   / |
    #  | ( )  |
    #  |   \ /
    #  \-- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((3, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target]*3 + [nid+1]*2 + [nid],
            [nid,nid+1,nid+2, nid,nid+2,  nid+2]
        ])
        update(g, new_x, new_edges)
        return self.target


class G24(Action):
    # ( ) -- (X) -- ( )
    #  |\   / |
    #  | ( )  |
    #  |   \  /
    #  \-- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target]*4 +        [nid,nid+1,nid+2],
            [nid,nid+1,nid+2,nid+3,   nid+1,nid+2,nid]
        ])
        update(g, new_x, new_edges)
        return self.target


class G25(Action):
    # ( ) -- (X) -- ( )
    #  |\   / |     /
    #  | ( )  |    /
    #  |   \  /   /
    #  \-- ( )---/
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target]*4 +        [nid,nid+1,nid+2,nid+2],
            [nid,nid+1,nid+2,nid+3,   nid+1,nid+2,nid,nid+3]
        ])
        update(g, new_x, new_edges)
        return self.target

class G26(Action):
    #       (X)
    #     / /    \
    #    /  |     \
    # ( ) --)----- ( )
    #   \  /      /
    #    ( ) -- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,nid,nid+1,nid+2,nid+3,  self.target,nid],
            [nid,nid+1,nid+2,nid+3,self.target,  nid+1,    nid+3]
        ])
        update(g, new_x, new_edges)
        return self.target


class G27(Action):
    #       (X)
    #     / / \  \
    #    /  | |   \
    # ( ) --)-)--- ( )
    #   \  /   \   /
    #    ( ) -- ( )
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,nid,nid+1,nid+2,nid+3,  self.target,self.target,nid],
            [nid,nid+1,nid+2,nid+3,self.target,  nid+1,      nid+2,    nid+3]
        ])
        update(g, new_x, new_edges)
        return self.target


class G28(Action):
    #        (X)
    #      / / \  \
    #     /  | |   \
    # -( ) --)-)--- ( )
    # |   \  /   \   /
    # |    ( ) -- ( )
    # \------------|
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,nid,nid+1,nid+2,nid+3,  self.target,self.target,nid,nid],
            [nid,nid+1,nid+2,nid+3,self.target,  nid+1,      nid+2,    nid+3,nid+2]
        ])
        update(g, new_x, new_edges)
        return self.target


class G29(Action):
    # Complete graph
    def execute(self, g):
        nid = g.x.size(0)
        new_x = torch.zeros((4, g.x.size(1)))
        new_x[:, self.ntype] = 1

        new_edges = torch.tensor([
            [self.target,nid,nid+1,nid+2,nid+3,  self.target,self.target,nid,nid,nid+1],
            [nid,nid+1,nid+2,nid+3,self.target,  nid+1,      nid+2,    nid+3,nid+2,nid+3]
        ])
        update(g, new_x, new_edges)
        return self.target

# How many nodes each action adds
N1 = [G1]
N2 = [G2, G6]
N3 = [G3, G5, G7, G9, G11, G23]
N4 = [G4, G8, G10, G12, G13, G14, G15, G16, G17, G18,
      G19, G20, G21, G22, G24, G25, G26, G27, G28, G29]