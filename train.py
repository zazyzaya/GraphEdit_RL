from types import SimpleNamespace

from joblib import Parallel, delayed
import torch

from environment.env import AStarEnv
from models.node_mapper import PPOModel, PPOMemory

GRAPH_SIZE = 10
HP = SimpleNamespace(
    epochs=1000,
    eps_per_update=10,
    workers=1,
    bs=1000
)

@torch.no_grad()
def generate_episode(model: PPOModel):
    env = AStarEnv(target_n=GRAPH_SIZE)
    s = env.state()
    halt = False
    buffer = PPOMemory(bs=1)

    while not halt:
        a, p, v = model.get_action(s)
        src = s.src[a[0]].item()
        dst = s.dst[a[1]].item() - s.offset

        next_s, r, halt = env.step(src,dst)
        buffer.remember(s, a, v, p, r, halt)
        s = next_s

    return buffer

def train(model, hp):
    for e in range(hp.epochs):
        mems = Parallel(n_jobs=hp.workers, prefer='processes')(
            delayed(generate_episode)(model)
            for _ in range(hp.eps_per_update)
        )
        avg_r = [sum(mem.r) for mem in mems]
        avg_r = sum(avg_r) / len(avg_r)

        buff = PPOMemory(hp.bs).load(mems)
        model.memory = buff
        model.learn()

        print(f'[{e+1}] Average reward: {avg_r}')

torch.autograd.set_detect_anomaly(True)
model = PPOModel(in_dim=5, hidden=256, lr=0.00003, epochs=5)
train(model, HP)