import torch
from torch import nn
from torch_geometric.nn.models import GIN
from torch_geometric.data import Data

from utils import pack_and_pad

class DQN(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()

        self.emb = GIN(in_dim, hidden, 3, hidden, jk='cat')

        '''
        self.combine = nn.Sequential(
            nn.Linear(hidden*3 + in_dim, hidden*2),
            nn.ReLU(),
            nn.Linear(hidden*2, hidden)
        )
        '''

    def forward(self, g: Data):
        z = self.emb(g.x, g.edge_index)
        z = torch.cat([g.x, z], dim=1)
        #z = self.combine(z)

        # Normalize so dot prod max is 1 min is 0
        norm = torch.norm(z, p=2, dim=1)
        z = z / norm

        src_z,s_mask = pack_and_pad(z[g.src], g.n_src, batch_first=True) # B x n_src x d
        dst_z,d_mask = pack_and_pad(z[g.dst], g.n_dst, batch_first=True) # B x n_dst x d

        # Need to explicitly repeat along columns
        d_mask = d_mask.repeat_interleave(z.size(1), dim=0).reshape(z.size())

        sim = 1-torch.cdist(src_z, dst_z) # B x s_max x d_max
        sim[s_mask] = -1
        sim[d_mask] = -1

        # TODO similarity does not equal value.
        # Can we do something about this?
        return sim.reshape(sim.size(0), -1)

    def get_action(self, g: Data) :
        values = self.forward(g)
