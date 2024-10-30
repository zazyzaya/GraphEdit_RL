import torch
from torch_geometric.data import Data

def num_to_batch(nums):
    batches = [0]
    for n in nums:
        batches.append(n.item() + batches[-1])
    return torch.tensor(batches)

def pack_and_pad(x, batches, batch_first=False):
    if not batch_first:
        return series_first_pack_and_pad(x,batches)
    else:
        return batch_first_pack_and_pad(x,batches)

def series_first_pack_and_pad(x, batches):
    largest = (batches[1:] - batches[:-1]).max()
    out = torch.zeros((largest, batches.size(0)-1, x.size(1)), device=x.device)
    mask = torch.zeros((batches.size(0)-1, largest), dtype=torch.bool, device=x.device)

    for i in range(batches.size(0)-1):
        st = batches[i]; en = batches[i+1]
        out[torch.arange(en-st), i] = x[st:en]
        mask[i][en-st:] = True

    return out,mask

def batch_first_pack_and_pad(x, batches):
    largest = (batches[1:] - batches[:-1]).max()
    out = torch.zeros((batches.size(0)-1, largest, x.size(1)), device=x.device)
    mask = torch.zeros((batches.size(0)-1, largest), dtype=torch.bool, device=x.device)

    for i in range(batches.size(0)-1):
        st = batches[i]; en=batches[i+1]
        out[i, torch.arange(en-st)] = x[st:en]
        mask[i][en-st:] = True

    return out,mask

def combine_subgraphs(gs):
    '''
    Input: list of graphs with fields
        g.x:    node colors
        g.edge_index:   edges
        g.src:  nodes that can be mapped to destination graph
        g.dst:  nodes that can have source nodes mapped to them

    Output:
        g.x
        g.edge_index
        g.src
        g.dst
        g.n_src: number of source nodes in each batch
        g.n_dst: number of destination nodes in each batch
        g.batches: CSR list of where combined graphs begin and end
    '''
    x = []
    ei = []
    src,n_src = [],[]
    dst,n_dst = [],[]
    batches = [0]

    offset = 0
    for g in gs:
        x.append(g.x)
        ei.append(g.edge_index + offset)
        batches.append(batches[-1] + g.x.size(0))

        src.append(g.src + offset)
        n_src.append(g.src.size(0))

        dst.append(g.dst + offset)
        n_dst.append(g.dst.size(0))

        offset += g.x.size(0)

    return Data(
        x=torch.cat(x),
        edge_index = torch.cat(ei, dim=1),
        batches=torch.tensor(batches),
        src = torch.cat(src),
        dst = torch.cat(dst),
        n_src = torch.tensor(n_src),
        n_dst = torch.tensor(n_dst)
    )