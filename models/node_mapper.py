import torch 
from torch import nn 
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.nn.models import GraphSAGE

def pack_and_pad(x, batches):
    largest = (batches[1:] - batches[:-1]).max()
    out = torch.empty((largest, batches.size(0)-1, x.size(1)), device=x.device)
    mask = torch.zeros((batches.size(0)-1, largest), dtype=torch.bool, device=x.device)

    for i in range(1,batches.size(0)): 
        st = batches[i]; en = batches[i-1]
        out[torch.arange(en-st), i] = x[st:en]
        mask[i][en-st:] = True 

    return out,mask

class Embedder(nn.Module): 
    def __init__(self, n_colors, emb_dim, khops=3, heads=8, trans_layers=4, trans_dim=512, device='cpu'): 
        super().__init__() 

        self.sage = GraphSAGE(n_colors, emb_dim, khops, emb_dim, droput=0.1, device=device)
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, heads, dim_feedforward=trans_dim, device=device),
            num_layers=trans_layers
        )
    
    def forward(self, g: Data): 
        '''
        Expects g to have the following keys: 
            x: |V| x n_colors  feature matrix 
            edge_index: 2 x |E| edge index 
            batches: CSR-style pointer to beginning and end of nodes in each graph 
                     E.g. 3 graphs with 3,4, and 5 nodes would be [0,3,7,13]
            (Model needs other keys for actor and critic)
        '''
        z = self.sage(g.x, g.edge_index) 
        z,mask = pack_and_pad(z, g.batches)
        return self.attn(z, src_key_padding_mask=mask)
    
class Actor(nn.Module): 
    def __init__(self, in_dim, hidden, layers=4):
         super().__init__()

         self.net = nn.Sequential(
             nn.Linear(in_dim, hidden), 
             nn.ReLU(), 
             *[
                 nn.Sequential(
                    nn.Linear(hidden//(2 ** i), hidden // (2 ** (i+1))), 
                    nn.ReLU()      
                 )
                 for i in range(1, layers)
             ]
         )

    def forward(self, z, src, dst, n_src, n_dst): 
        '''
        z:  B x d  input 
        src: Source node indexes (padded) 
        dst: Dst node indexes    (padded)
        n_src: B-dim list of num src nodes in batch
        n_dst: B-dim list of num dst nodes in batch 
        '''
        
        b = n_src.size(0)

        n_srcs_padded = src.size(0)
        src = self.net(z[src])
        src = src.view(b, src.size(0) // b, src.size(-1))

        dst = self.net(z[dst])
        dst = dst.view(b, dst.size(0) // b, dst.size(-1))
        
        # Dot products of all src-dst combos
        probs = src @ dst.transpose(1,2)

        # Mask out rows/cols coresponding to masked nodes
        # that don't matter but had to be padded in 
        for i in range(b): 
            probs[i][n_src[i]-1:] = float('-inf')
            probs[i][:, n_dst[i]-1:] = float('-inf')

        # Flatten to B x max_src*max_dst 
        probs = probs.view(
            probs.size(0), 
            probs.size(1) * probs.size(2)
        )

        return Categorical(logits=probs)