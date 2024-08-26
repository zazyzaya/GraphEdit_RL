import torch
from torch.distributions import Categorical
from torch import nn
from torch.optim.adam import Adam

from models.env_repr import GraphEmbedder, pack_sequence
from environment.state import batch_states

class PPOMemory:
    def __init__(self, bs):
        self.s = []
        self.a = []
        self.v = []
        self.p = []
        self.r = []
        self.t = []

        self.bs = bs

    def load(self, mems):
        for mem in mems:
            self.s += mem.s; self.a += mem.a
            self.v += mem.v; self.p += mem.p
            self.r += mem.r; self.t += mem.t

        return self

    def remember(self, s,a,v,p,r,t):
        self.s.append(s)
        self.a.append(a)
        self.v.append(v)
        self.p.append(p)
        self.r.append(r)
        self.t.append(t)

    def clear(self):
        self.s = []; self.a = []
        self.v = []; self.p = []
        self.r = []; self.t = []

    def get_batches(self):
        idxs = torch.randperm(len(self.a))
        batch_idxs = idxs.split(self.bs)

        return self.s, self.a, self.v, \
            self.p, self.r, self.t, batch_idxs

    def combine(self, others):
        for o in others:
            self.s += o.s; self.a += o.a
            self.v += o.v; self.p += o.p
            self.r += o.r; self.t += o.t

        return self


class Actor(nn.Module):
    def __init__(self, hidden, action_embs, latent=16):
        super().__init__()

        self.action_selection = nn.Sequential(
            nn.Linear(action_embs.size(1)+hidden+hidden, hidden*2),
            nn.ReLU(),
            nn.Linear(hidden*2, latent)
        )
        self.node_selection = nn.Sequential(
            nn.Linear(hidden+hidden+hidden, hidden*2),
            nn.ReLU(),
            nn.Linear(hidden*2, latent)
        )
        self.action_embs = action_embs

    def forward(self, z,batch_sizes, state_g, target_g):
        '''
        z:          V x d_z         matrix of node embeddings for the input graphs
        batch_sizes:
                    |G| x 1         list of graph sizes for batched graphs
        state_g:    B x d_g         Global state vector for each graph
        target_g:   B x d_g         Global state vector for each target graph
        '''
        batch_size = state_g.size(0)
        n_actions = self.action_embs.size(0)

        if batch_size > 1:
            ae = self.action_embs.repeat(batch_size,1)      # B*A x d_a
        else:
            ae = self.action_embs

        sga = state_g.repeat_interleave(n_actions,dim=0)    # B*A x d_g
        tga = target_g.repeat_interleave(n_actions,dim=0)   # B*A x d_g

        a_query = torch.cat([ae, sga, tga], dim=1)
        action_probs = self.action_selection(a_query)               # B*A x d
        action_probs = action_probs.reshape(                        # B x A x d
            state_g.size(0),
            self.action_embs.size(0),
            action_probs.size(-1)
        )

        sgn = torch.repeat_interleave(state_g, batch_sizes, dim=0)
        tgn = torch.repeat_interleave(target_g, batch_sizes, dim=0)
        z_query = torch.cat([z, sgn, tgn], dim=1)

        node_probs = self.node_selection(z_query)
        node_probs,mask = pack_sequence(node_probs, batch_sizes) # B x V_max x d

        # Combine prob of node selection with prob of action selection to find
        # P(N=n, A=a) (not necessarilly independant)
        probs = node_probs @ action_probs.transpose(1,2)        # B x V_max x A
        probs[mask.squeeze(-1) == 0] = float('-inf')

        # Reshape s.t. each row is a batch, and values are
        # [a0n0, a1n0, a2n0, ... akn0, a0n1, ..., akn1, ..., aknj]
        probs = probs.reshape(probs.size(0), probs.size(1)*probs.size(2)) # B x V*A
        return Categorical(logits=probs)

class Critic(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden*4),
            nn.ReLU(),
            nn.Linear(hidden*4, hidden*2),
            nn.ReLU(),
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, g_state, g_target):
        x = torch.cat([g_state, g_target], dim=1)
        return self.net(x)

class GraphPPO(nn.Module):
    def __init__(self, in_dim, hidden, action_embs,
                 gamma=0.99, lmbda=0.95, clip=0.1, bs=1000, epochs=10, lr=1e-3):
        super().__init__()
        self.gamma = gamma
        self.lmbda = lmbda
        self.clip = clip
        self.bs = bs
        self.epochs = epochs

        self.args = (in_dim, hidden, action_embs)
        self.kwargs = dict(gamma=gamma, lmbda=lmbda, clip=clip, bs=bs, epochs=epochs, lr=lr)
        self.mse = nn.MSELoss()

        self.memory = PPOMemory(self.bs)
        self.env_repr = GraphEmbedder(in_dim, hidden, hidden)
        self.actor = Actor(hidden, action_embs)
        self.critic = Critic(hidden*2, hidden)
        self.opt = Adam(self.parameters(), lr=lr)
        self.deterministic = False

    @torch.no_grad()
    def get_action_batched(self, state_g, state_batch_sizes, target_g, target_batch_sizes):
        state_z,state_g = self.env_repr.forward(state_g, state_batch_sizes)
        _,target_g = self.env_repr.forward(target_g, target_batch_sizes)
        distro = self.actor(state_z,state_batch_sizes, state_g,target_g)

        # I don't know why this would ever be called
        # during training, but just in case, putting the
        # logic block outside the training check
        if self.deterministic:
            action = distro.probs.argmax()
        else:
            action = distro.sample()

        if not self.training:
            return action.item()

        value = self.critic(state_g, target_g)
        prob = distro.log_prob(action)
        return action.item(), value.item(), prob.item()

    @torch.no_grad
    def get_action(self, state_g, target_g):
        '''
        Default code wants to get actions for large batches of states
        But if you only have one graph, you dont need the code to get
        batch sizes and combine disjoint graphs, etc.
        This just makes it a bit easier to call get_action during inference
        '''
        state_batch_sizes = torch.tensor([state_g.x.size(0)])
        target_batch_sizes = torch.tensor([target_g.x.size(0)])
        return self.get_action_batched(state_g, state_batch_sizes, target_g, target_batch_sizes)

    @torch.no_grad
    def take_action(self, state_g, target_g, action):
        state_batch_sizes = torch.tensor([state_g.x.size(0)])
        target_batch_sizes = torch.tensor([target_g.x.size(0)])

        state_z,state_g = self.env_repr.forward(state_g, state_batch_sizes)
        _,target_g = self.env_repr.forward(target_g, target_batch_sizes)
        distro = self.actor(state_z,state_batch_sizes, state_g,target_g)

        value = self.critic(state_g, target_g)
        prob = distro.log_prob(action)
        return action.item(), value.item(), prob.item()

    def learn(self, verbose=True):
        '''
        Assume that an external process is adding memories to
        the PPOMemory unit, and this is called every so often
        '''
        for e in range(self.epochs):
            s,a,v,p,r,t, batches = self.memory.get_batches()

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

            for b in batches:
                b = b.tolist()
                new_probs = []

                s_ = [s[idx] for idx in b]
                a_ = [a[idx] for idx in b]
                state_g,state_batch_sizes, target_g,target_batch_sizes = batch_states(s_)

                state_z,state_g = self.env_repr.forward(state_g, state_batch_sizes)
                _,target_g = self.env_repr.forward(target_g, target_batch_sizes)
                dist = self.actor(state_z, state_batch_sizes, state_g, target_g)
                critic_vals = self.critic(state_g, target_g)

                new_probs = dist.log_prob(torch.tensor(a_))
                old_probs = torch.tensor([p[i] for i in b])
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
                    print(f'[{e}] C-Loss: {0.5*critic_loss.item():0.4f}  A-Loss: {actor_loss.item():0.4f} E-loss: {-entropy_loss.item()*0.01:0.4f}')

        # After we have sampled our minibatches e times, clear the memory buffer
        self.memory.clear()
        return total_loss.item()

    def save(self, outf='ppo.pt'):
        sd = self.state_dict()
        opt_state = self.opt.state_dict()

        torch.save({
            'args': self.args,
            'kwargs': self.kwargs,
            'sd': sd,
            'opt_sd': opt_state
        }, outf)

    def remember(self, s,a,v,p,r,t):
        self.memory.remember(s,a,v,p,r,t)


def load_ppo(fname):
    data = torch.load(fname)
    model = GraphPPO(*data['args'], **data['kwargs'])
    model.load_state_dict(data['sd'])
    model.opt.load_state_dict(data['opt_sd'])

    return model