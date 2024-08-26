import torch
from torch import nn
from torch_geometric.nn import SAGEConv

def pack_sequence(seq, lens):
    '''
    Given sequence of
    [(x1, x2), (x3, x4, x5)]
    (parens are implied. This is a continuous vec)

    and len of
    [2, 3]

    return [ [x1, x2, 0], [x3, x4, x5] ]

    '''
    padded = torch.zeros(lens.size(0), lens.max(), seq.size(-1))
    mask = torch.zeros(lens.size(0), lens.max(), 1)

    offset = 0
    for i,len in enumerate(lens):
        padded[i][0:len] = seq[offset : offset+len]
        mask[i][0:len] = 1
        offset += len

    return padded, mask

class SimpleSelfAttention(nn.Module):
    '''
    Implimenting global-node self-attention from
        https://arxiv.org/pdf/2009.12462.pdf
    '''
    def __init__(self, in_dim, h_dim, g_dim):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Softmax(dim=-1)
        )
        self.feat = nn.Linear(in_dim, h_dim)
        self.glb = nn.Sequential(
            nn.Linear(h_dim+g_dim, g_dim),
            nn.Tanh()
        )

        self.g_dim = g_dim
        self.h_dim = h_dim

    def forward(self, v, mask, g=None):
        '''
        Inputs:
            v:      B x N x d tensor
            mask:   B x N x 1 tensor of 1s or 0s
            g:      B x d tensor
        '''
        if g is None:
            g = torch.zeros((v.size(0), self.g_dim))

        att = self.att(v)                   # B x N x h
        feat = self.feat(v)                 # B x N x h
        out = (att*feat*mask).sum(dim=1)    # B x h

        g_ = self.glb(torch.cat([out,g], dim=-1))  # B x g
        return g + g_                              # Short-circuit


class GraphEmbedder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=3):
        super().__init__()

        self.layers = layers
        self.attn_layers = nn.ParameterList(
            [SimpleSelfAttention(in_dim, hidden, hidden)] +
            [
                SimpleSelfAttention(hidden, hidden, hidden)
                for _ in range(layers-1)
            ] +
            [SimpleSelfAttention(out_dim, hidden, hidden)]
        )

        self.gnn_layers = nn.ParameterList(
            [SAGEConv(in_dim+hidden, hidden, aggr='sum')] +
            [
                SAGEConv(hidden*2, hidden, aggr='sum')
                for _ in range(layers-2)
            ] +
            [SAGEConv(hidden*2, out_dim, aggr='sum')]
        )

    def _attn_forward(self, i,x,batch_sizes,attn):
        x,mask = pack_sequence(x, batch_sizes)
        return self.attn_layers[i](x, mask, g=attn)

    def _layer_forward(self, i, x,ei,batch_sizes, attn=None):
        attn = self._attn_forward(i, x,batch_sizes,attn)
        attn_resized = torch.repeat_interleave(attn, batch_sizes, dim=0)

        x = torch.cat([x, attn_resized], dim=1)
        return torch.relu(self.gnn_layers[i](x, ei)), attn

    def forward(self, g, batch_sizes):
        x = g.x
        ei = g.edge_index
        attn = None

        for i in range(self.layers):
            x, attn = self._layer_forward(i, x,ei, batch_sizes, attn=attn)

        attn = self._attn_forward(self.layers, x, batch_sizes, attn)
        #x,mask = pack_sequence(x, batch_sizes)
        return x,attn