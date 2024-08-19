import torch
from torch import nn
from torch_geometric.nn import SAGEConv

class HVAE(nn.Module):
    def __init__(self, n_acts, obs_dim, act_latent=32, obs_latent=64):
        super().__init__()
        def two_layer(in_dim, hidden, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim)
            )

        self.args = (n_acts, obs_dim)
        self.kwargs = dict(act_latent=act_latent, obs_latent=obs_latent)

        # Action encoder
        self.action = two_layer(n_acts, act_latent*4, act_latent*2)
        self.action_mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(act_latent*2, act_latent)
        ) # No final activation
        self.action_logvar = nn.Sequential(
            nn.ReLU(),
            nn.Linear(act_latent*2, act_latent)
        ) # No final activation

        # ObservationGraph embedder
        self.gnn_1 = SAGEConv(obs_dim, obs_latent*4, aggr='sum')
        self.gnn_2 = SAGEConv(obs_latent*4, obs_latent*2, aggr='sum')
        self.gnn_3 = SAGEConv(obs_latent*2, obs_latent, aggr='sum')

        # Observation encoder
        self.graph_obs = two_layer(obs_latent+act_latent, obs_latent*4, obs_latent*2)
        self.graph_mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(obs_latent*2, obs_latent)
        )
        self.graph_logvar = nn.Sequential(
            nn.ReLU(),
            nn.Linear(obs_latent*2, obs_latent)
        )

        # Predicting how GNN emb changes between two runs
        self.decoder = nn.Sequential(
            two_layer(obs_latent+act_latent, obs_latent*2, obs_latent*4),
            nn.ReLU(),
            two_layer(obs_latent*4, obs_latent*2, obs_latent)
        )

    def _reparam_trick(self, mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def _kl_loss(self, mu,logvar):
        '''
        Because KL(mu,logvar || N(0,1)) =
            -log(sigma) + (sigma^2 + mu^2 - 1)/2

            = (-2log(sigma) + sigma^2 + mu^2 - 1) / 2
            = - ( 1 + log(sigma^2) + sigma^2 + mu^2 ) / 2

            ( logvar = log(sigma^2) )
        '''
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def _graph_embed(self, g):
        z = torch.relu(self.gnn_1(g.x, g.edge_index))
        z = torch.relu(self.gnn_2(z, g.edge_index))
        z = torch.tanh(self.gnn_3(z, g.edge_index))
        return z

    def encode_action(self, act_vec):
        act_vec = self.action(act_vec)

        mu = self.action_mu(act_vec)
        logvar = torch.empty(mu.size()) # Only calcuated if needed

        # Only use variational embs during training
        if self.training:
            logvar = self.action_logvar(act_vec)
            act_z = self._reparam_trick(mu, logvar)
        else:
            act_z = mu

        return act_z, mu, logvar

    def encode_graph_obs(self, action_z, graph_z):
        graph_vec = torch.cat([action_z, graph_z], dim=1)
        graph_vec = self.graph_obs(graph_vec)

        mu = self.graph_mu(graph_vec)
        logvar = torch.empty(mu.size())

        if self.training:
            logvar = self.graph_logvar(graph_vec)
            graph_z = self._reparam_trick(mu, logvar)
        else:
            graph_z = self.graph_mu(graph_vec)

        return graph_z, mu, logvar

    def forward(self, act_vec, before_id, after_id, g_sequence):
        z_nodes = self._graph_embed(g_sequence)

        # So id -1 just returns zero vec
        z_nodes = torch.cat([z_nodes, torch.zeros(1,z_nodes.size(1))], dim=0)

        z_before = z_nodes[before_id]
        z_after = z_nodes[after_id]

        action_z, a_mu, a_logvar = self.encode_action(act_vec)
        graph_z, g_mu, g_logvar = self.encode_graph_obs(action_z, z_before)
        decoded = self.decoder(torch.cat([action_z, graph_z], dim=1))

        recon_loss = (decoded - z_after).pow(2).mean()
        a_kl_loss = self._kl_loss(a_mu, a_logvar)
        g_kl_loss = self._kl_loss(g_mu, g_logvar)

        return recon_loss, a_kl_loss, g_kl_loss


    def save(self, fname='action_enc.pt'):
        torch.save(
            (self.args, self.kwargs, self.state_dict()),
            fname
        )
