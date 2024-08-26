from types import SimpleNamespace

import torch
from torch.optim.adam import Adam

from generator import generate_rand_episode
from models.action_encoder import HVAE

HP = SimpleNamespace(
    lr = 1e-3,
    epochs = 2_500,
    batch_size = 250,
    latent_a = 64,
    latent_g = 64
)

def train(model: HVAE):
    opt = Adam(model.parameters(), lr=HP.lr)

    for e in range(HP.epochs):
        opt.zero_grad()
        (before,after,g), action_vecs = generate_rand_episode(HP.batch_size)
        rloss,akl,gkl = model.forward(action_vecs, before, after, g)
        loss = rloss + akl + gkl
        loss.backward()
        opt.step()

        if e % 25 == 0:
            print(f'[{e}] {loss.item():0.4f} (R: {rloss.item():0.4f}, A-KL: {akl.item():0.4f}, G-KL: {gkl.item():0.4f})')

        if e % 1_000 == 999:
            model.save()

    model.save()


if __name__ == '__main__':
    (_,_,g), a = generate_rand_episode(3)
    model = HVAE(
        a.size(1), g.x.size(1),
        act_latent=HP.latent_a, obs_latent=HP.latent_g
    )

    train(model)