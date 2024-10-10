import random

import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_undirected, add_remaining_self_loops
from torch_geometric.nn import MessagePassing
from tqdm import tqdm


N_COLORS = 5
HOMOPHILY = 0.5 # Percent to be filled in from neighbors color

MP = MessagePassing(aggr='max')

def batch_graphs(targets,post_targets, xs, eis):
    sizes = [x.size(0) for x in xs]

    offset = 0
    for i in range(len(xs)):
        xs[i] += offset
        eis[i] += offset

        if i != len(xs)-1:
            targets[i] += offset
        if i:
            post_targets[i-1] += offset

        offset += sizes[i]

    targets[targets.isnan()] = -1
    post_targets[post_targets.isnan()] = -1

    return targets.long(), post_targets.long(), Data(
        x = torch.cat(xs, dim=0),
        edge_index = torch.cat(eis, dim=1),
        sizes = torch.tensor(sizes)
    )

def generate_sample(n, p, n_colors=N_COLORS, homophily=HOMOPHILY):
    ei = erdos_renyi_graph(n,p)
    ei = to_undirected(ei.unique(dim=1))
    ei = add_remaining_self_loops(ei)[0]
    x = torch.zeros(n, n_colors)

    # Color some percentage randomly
    rnd_color = torch.rand(n) > homophily
    rnd_colors = torch.randint(0, n_colors, (rnd_color.sum(),))
    x[rnd_color, rnd_colors] = 1

    # Color the rest based on their neighbors color
    uncolored = x.sum(dim=1) == 0
    while uncolored.sum():
        msg = MP.propagate(ei, size=None, x=x)
        x[uncolored] = msg[uncolored]

        # If node has more than one color, select
        # one randomly
        multicolored = x.sum(dim=1) > 1
        color = torch.multinomial(x[multicolored], 1).flatten()
        x[multicolored] = 0
        x[multicolored, color] = 1

        # Color more uncolored nodes randomly
        uncolored = x.sum(dim=1) == 0
        n = (uncolored).sum().item()

        if n == 0:
            break

        rnd_color = torch.rand(n) > homophily
        rnd_colors = torch.randint(0, n_colors, (rnd_color.sum(),))
        x[uncolored.nonzero()[rnd_color].flatten(), rnd_colors] = 1

        # Loop again
        uncolored = x.sum(dim=1) == 0

    # Flatten colors and remove self-loops 
    x = x.nonzero()[:,1]
    ei = ei[:, ei[0] != ei[1]]

    return Data(x=x, edge_index=ei)