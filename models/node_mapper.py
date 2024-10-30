import torch
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.nn.models import GraphSAGE
from torch.optim import Adam

from models.memory_buffer import PPOMemory
from models.utils import combine_subgraphs, num_to_batch, pack_and_pad


class Embedder(nn.Module):
    def __init__(self, n_colors, emb_dim, khops=3, heads=8, trans_layers=4, trans_dim=512, device='cpu'):
        super().__init__()

        self.sage = GraphSAGE(n_colors, emb_dim, khops, emb_dim, dropout=0.1)
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
        z = self.sage.forward(g.x, g.edge_index)
        z,mask = pack_and_pad(z, g.batches)
        z = self.attn(z, src_key_padding_mask=mask)

        return z[~mask.T]

class Actor(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()

        def get_net():
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden//2),
                nn.ReLU(),
            )

        self.src_net = get_net()
        self.dst_net = get_net()
        self.sig = nn.Sigmoid()

    def forward(self, z, src, dst, n_src, n_dst):
        '''
        z:  B x d  input
        src: Source node indexes (padded)
        dst: Dst node indexes    (padded)
        n_src: B-dim list of num src nodes in batch
        n_dst: B-dim list of num dst nodes in batch
        '''
        src = self.src_net(z[src])
        #src = src / (torch.norm(src, p=2, dim=-1, keepdim=True) + 1e-8)
        src_batch = num_to_batch(n_src)
        src,src_mask = pack_and_pad(src, src_batch, batch_first=True)

        dst = self.dst_net(z[dst])
        #dst = dst / (torch.norm(dst, p=2, dim=-1, keepdim=True) + 1e-8)
        dst_batch = num_to_batch(n_dst)
        dst,dst_mask = pack_and_pad(dst, dst_batch, batch_first=True)

        # Dot products of all src-dst combos
        if src.isnan().any() or dst.isnan().any():
            print("Hm")

        probs = src @ dst.transpose(1,2)

        if probs.isnan().any():
            print("Hm")

        # Mask out rows/cols coresponding to masked nodes
        # that don't matter but had to be padded in
        for i in range(probs.size(0)):
            probs[i][n_dst[i]:] = float('-inf')
            probs[i][:, n_src[i]:] = float('-inf')

        # Flatten to B x max_src*max_dst
        probs = probs.view(probs.size(0), -1)

        return Categorical(logits=probs)


class Critic(nn.Module):
    '''
    Could be simplified, but this is how we will be doing
    A* later on, so should be as powerful as Actor module
    '''
    def __init__(self, in_dim, hidden):
        super().__init__()

        self.src_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.dst_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        '''
        # Combine src and dst for initial attention
        self.src_dst_attn = nn.MultiheadAttention(hidden//2, 8)

        self.src_kv = nn.Sequential(
            nn.Linear(hidden//2, hidden),
            nn.ReLU()
        )

        # Then combine output of above with single parameter for B x 1 x d output
        self.out_attn = nn.MultiheadAttention(hidden//2, 8)
        self.out_q = nn.Parameter(torch.empty(1,1,hidden//2))
        torch.nn.init.xavier_normal_(self.out_q)
        '''

        # Finally, project into single dimension for output
        self.out = nn.Linear(hidden // 2, 1)

    def forward(self, z, src, dst, n_src, n_dst):
        '''
        z:  B x d  input
        src: Source node indexes (padded)
        dst: Dst node indexes    (padded)
        n_src: B-dim list of num src nodes in batch
        n_dst: B-dim list of num dst nodes in batch
        '''
        src = self.src_net(z[src])
        #src = src / (torch.norm(src, p=2, dim=-1, keepdim=True) + 1e-8)
        src_batch = num_to_batch(n_src)
        src,src_mask = pack_and_pad(src, src_batch, batch_first=True)

        dst = self.dst_net(z[dst])
        #dst = dst / (torch.norm(dst, p=2, dim=-1, keepdim=True) + 1e-8)
        dst_batch = num_to_batch(n_dst)
        dst,dst_mask = pack_and_pad(dst, dst_batch, batch_first=True)

        # Dot products of all src-dst combos
        if src.isnan().any() or dst.isnan().any():
            print("Hm")

        probs = src @ dst.transpose(1,2)

        if probs.isnan().any():
            print("Hm")

        # Mask out rows/cols coresponding to masked nodes
        # that don't matter but had to be padded in
        outs = torch.zeros(probs.size(0), 1)
        for i in range(probs.size(0)):
            probs[i][n_dst[i]:] = float('-inf')
            max_dst = probs[i].max(dim=-1).values
            outs[i] = max_dst[:n_src[i]].mean()

        return outs


class PPOModel(nn.Module):
    def __init__(self, in_dim, hidden, k_hops=3, layers=4, batch_size=2048,
                 lr=0.001, epochs=10, gamma=0.99, clip=0.1):
        super().__init__()

        # Submodules
        self.emb = Embedder(in_dim, hidden, khops=k_hops)
        self.actor = Actor(hidden, hidden)
        self.critic = Critic(hidden, hidden)

        # PPO Learning params
        self.opt = Adam(self.parameters(), lr=lr)
        self.epochs = epochs
        self.gamma = gamma
        self.clip = clip
        self.mse = nn.MSELoss()

        self.memory = PPOMemory(bs=batch_size)

    def forward(self, g):
        '''
        Expects g to have the following keys:
            x: |V| x n_colors  feature matrix
            edge_index: 2 x |E| edge index
            batches: CSR-style pointer to beginning and end of nodes in each graph
                     E.g. 3 graphs with 3,4, and 5 nodes would be [0,3,7,13]
            src: Source node indexes (padded)
            dst: Dst node indexes    (padded)
            n_src: B-dim list of num src nodes in batch
            n_dst: B-dim list of num dst nodes in batch
        '''

        z = self.emb(g)

        if z.isnan().any():
            print("hm")

        probs = self.actor(z, g.src, g.dst, g.n_src, g.n_dst)
        value = self.critic(z, g.src, g.dst, g.n_src, g.n_dst)

        return probs, value

    @torch.no_grad()
    def get_action(self, g):
        probs, value = self.forward(g)
        a = probs.sample()
        a_ = self.__prob_idx_to_action(a, g.n_dst.max())
        return a_, probs.log_prob(a).item(), value.item()

    def __prob_idx_to_action(self, actions, max_dst_nodes):
        # Represent source to destination mapping 2 x B matrix of edges
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        src_nodes = actions // max_dst_nodes
        dst_nodes = actions % max_dst_nodes
        return torch.cat([src_nodes.T, dst_nodes.T], dim=0)

    def __action_to_prob_idx(self, actions, max_dst_nodes):
        # Represent 2 x B edge matrix as B x 1 action prob index
        a = actions[0] * max_dst_nodes
        a += actions[1]
        return a

    def learn(self, verbose=True):
        '''
        Assume that an external process is adding memories to
        the PPOMemory unit, and this is called every so often
        '''
        for e in range(self.epochs):
            s,a,v,p,r,t, batches = self.memory.get_batches()

            '''
            advantage = torch.zeros(len(s), dtype=torch.float)

            # Probably a more efficient way to do this in parallel w torch
            for t in range(len(s)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(s)-1):
                    a_t += discount*(r[k] + self.gamma*v[k+1] -v[k])
                    discount *= self.gamma*self.lmbda

                advantage[t] = a_t
            '''
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(r), reversed(t)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + self.gamma * discounted_reward
                rewards.insert(0, discounted_reward)

            r = torch.tensor(rewards, dtype=torch.float)
            r = (r - r.mean()) / (r.std() + 1e-5) # Normalize rewards

            advantages = r - torch.tensor(v)

            for b_idx,b in enumerate(batches):
                b = b.tolist()
                new_probs = []

                # Evaluate previously seen states
                s_ = [s[idx] for idx in b]
                g = combine_subgraphs(s_)
                dist, critic_vals = self.forward(g)

                # Batch actions
                a_ = [a[idx] for idx in b]
                a_old = torch.concat(a_, dim=1)
                a_ = self.__action_to_prob_idx(a_old, g.n_dst.max())

                # Compare old and new action probs
                new_probs = dist.log_prob(a_)
                old_probs = torch.tensor([[p[i]] for i in b])
                entropy = dist.entropy()

                a_t = advantages[b]

                # Equiv to exp(new) / exp(old) b.c. recall: these are log probs
                r_theta = (new_probs - old_probs).exp()
                clipped_r_theta = torch.clip(
                    r_theta, min=1-self.clip, max=1+self.clip
                )

                # Use whichever one is minimal
                actor_loss = torch.min(r_theta*a_t, clipped_r_theta*a_t)
                actor_loss = -actor_loss.mean()

                # Critic uses MSE loss between expected value of state and observed
                # reward with discount factor
                critic_loss = self.mse(r[b].unsqueeze(-1), critic_vals)

                # Not totally necessary but maybe will help?
                entropy_loss = entropy.mean()

                # Calculate gradient and backprop
                total_loss = actor_loss + 0.5*critic_loss - 0.01*entropy_loss
                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()

                if verbose:
                    print(f'\t[{e}] C-Loss: {0.5*critic_loss.item():0.4f}  A-Loss: {actor_loss.item():0.4f} E-loss: {-entropy_loss.item()*0.01:0.4f}')

        # After we have sampled our minibatches e times, clear the memory buffer
        self.memory.clear()
        return total_loss.item()