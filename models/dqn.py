import torch
from torch import nn
from torch_geometric.nn.models import GIN
from torch_geometric.data import Data

from models.utils import pack_and_pad, num_to_batch

class DQN(nn.Module):
    def __init__(self, in_dim, hidden=32, eps=0.9):
        super().__init__()

        self.emb = GIN(in_dim, hidden, 3, hidden, jk='cat')

        self.out = nn.Sequential(
            nn.Linear(hidden + in_dim, hidden*2),
            nn.ReLU(),
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.eps = eps

    def forward_distance_measure(self, g: Data):
        z = self.emb(g.x, g.edge_index)
        z = torch.cat([g.x, z], dim=1)
        #z = self.combine(z)

        # Normalize so dot prod max is 1 min is 0
        norm = torch.norm(z, p=2, dim=1)
        z = z / norm

        src_z,s_mask = pack_and_pad(z[g.src], num_to_batch(g.n_src), batch_first=True) # B x n_src x d
        dst_z,d_mask = pack_and_pad(z[g.dst], num_to_batch(g.n_dst), batch_first=True) # B x n_dst x d

        # Need to explicitly repeat along columns
        d_mask = d_mask.repeat_interleave(z.size(1), dim=0).reshape(z.size())

        sim = 1-torch.cdist(src_z, dst_z) # B x s_max x d_max
        sim[s_mask] = -1
        sim[d_mask] = -1

        # TODO similarity does not equal value.
        # Can we do something about this?
        return sim.reshape(sim.size(0), -1)

    def forward(self, g: Data):
        z = self.emb(g.x, g.edge_index)
        z = torch.cat([g.x, z], dim=1)
        #z = self.combine(z)

        # Normalize so dot prod max is 1 min is 0
        norm = torch.norm(z, p=2, dim=1, keepdim=True)
        z = z / norm

        src_z,s_mask = pack_and_pad(z[g.src], num_to_batch(g.n_src), batch_first=True) # B x n_src x d
        dst_z,d_mask = pack_and_pad(z[g.dst], num_to_batch(g.n_dst), batch_first=True) # B x n_dst x d

        max_src = src_z.size(1)
        max_dst = dst_z.size(1)

        src_z = src_z.repeat_interleave(max_dst, dim=1)
        s_mask = s_mask.repeat_interleave(max_dst, dim=-1)
        dst_z = dst_z.repeat(1,max_src,1)
        d_mask = d_mask.repeat(1,max_src)

        comb_z = src_z * dst_z
        value = self.out(comb_z)
        value[s_mask] = float('-inf')
        value[d_mask] = float('-inf')

        return value.squeeze(-1), max_src, max_dst


    def get_action(self, g: Data) :
        values, max_src, max_dst = self.forward(g)
        rnd = torch.rand(values.size(0))

        val,idx = values.max(dim=-1)
        r_mask = (rnd > self.eps).nonzero().squeeze(-1)

        if r_mask.size(0):
            # Select any action other than illegal ones
            rnd_idx = torch.multinomial((values[r_mask] != float('-inf')).float(), 1)
            rnd_val = values[r_mask, rnd_idx]

            val[r_mask] = rnd_val
            idx[r_mask] = rnd_idx

        src = g.src[idx // max_dst].item()
        dst = (g.dst[idx % max_dst] - g.offset).item()

        return (src,dst), values
