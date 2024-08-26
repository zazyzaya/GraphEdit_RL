import torch
from torch_geometric.data import Data

def batch_states(states):
    cx, tx, ce, te = [],[],[],[]
    for s in states:
        cx.append(s[0].x.clone())
        tx.append(s[1].x.clone())
        ce.append(s[0].edge_index.clone())
        te.append(s[1].edge_index.clone())

    csizes,tsizes = [],[]
    coffset = toffset = 0
    for i in range(len(states)):
        ce[i] += coffset
        te[i] += toffset

        csizes.append(cx[i].size(0))
        tsizes.append(tx[i].size(0))

        coffset += csizes[-1]
        toffset += tsizes[-1]

    state_g = Data(
        x = torch.cat(cx, dim=0),
        edge_index = torch.cat(ce, dim=1),
    )
    target_g = Data(
        x = torch.cat(tx, dim=0),
        edge_index = torch.cat(te, dim=1)
    )
    return state_g, torch.tensor(csizes), target_g, torch.tensor(tsizes)