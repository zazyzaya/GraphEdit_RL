import random

import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

from environment.actions import ACTION_TO_IDX, NEEDS_COLOR, SIMPLE_ACTIONS, AddNode, N_ACTIONS
from environment.actions.graphlet_actions import N1, N2, N3, N4

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
        edge_index = torch.cat(eis, dim=1)
    )

def encode_actions(actions, params):
    act_ids = torch.zeros(len(actions), N_ACTIONS)
    param_vec = torch.zeros(len(actions), N_COLORS)

    act_ids[torch.arange(actions.size(0)), actions] = 1
    needs_param = actions.apply_(NEEDS_COLOR).bool()
    param_vec[needs_param, params[needs_param]] = 1

    return torch.cat([act_ids, param_vec], dim=1)


def generate_sample(n, p):
    g = erdos_renyi_graph(n,p)
    x = torch.zeros(n, N_COLORS)
    accepting_edges = torch.zeros(n, 1)

    # Color some percentage randomly
    rnd_color = torch.rand(n) > HOMOPHILY
    rnd_colors = torch.randint(0, N_COLORS, (rnd_color.sum(),))
    x[rnd_color, rnd_colors] = 1

    # Color the rest based on their neighbors color
    uncolored = x.sum(dim=1) == 0
    while uncolored.sum():
        msg = MP.propagate(g, size=None, x=x)
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

        rnd_color = torch.rand(n) > HOMOPHILY
        rnd_colors = torch.randint(0, N_COLORS, (rnd_color.sum(),))
        x[uncolored.nonzero()[rnd_color].flatten(), rnd_colors] = 1

        # Loop again
        uncolored = x.sum(dim=1) == 0

    return torch.cat([x,accepting_edges], dim=1), g

def generate_rand_episode(steps, n=50, p=0.1):
    xs,eis = [],[]

    x,ei = generate_sample(n, p)
    targets,post_targets,actions,params = [],[],[],[]
    for _ in range(steps):
        xs.append(x.clone())
        eis.append(ei.clone())

        action_type = random.choice([
            SIMPLE_ACTIONS, N1, N2, N3, N4
        ])

        act = random.choice(action_type)

        if act != AddNode:
            target = random.randint(0,x.size(0)-1)
        else:
            target = float('nan')

        color = random.randint(0,N_COLORS-1)

        post_targets.append(
            act(target=target, ntype=color).execute(Data(x=x,edge_index=ei))
        )
        targets.append(target)
        actions.append(ACTION_TO_IDX[act])
        params.append(color)

    xs.append(x.clone())
    eis.append(ei.clone())

    targets = torch.tensor(targets)
    post_targets = torch.tensor(post_targets)
    actions = torch.tensor(actions)
    params = torch.tensor(params)

    return \
        batch_graphs(targets,post_targets, xs, eis), \
        encode_actions(actions, params)


if __name__ == "__main__":
    generate_rand_episode(10_000)