from types import SimpleNamespace

from joblib import Parallel, delayed
import torch

from environment.env import AStarEnv
from environment.heuristic_agents import HeuristicAgent, GreedyAgent
from models.node_mapper import PPOModel, PPOMemory

torch.set_num_threads(32)

GRAPH_SIZE = 30
HP = SimpleNamespace(
    epochs=1000,
    eps_per_update=100,
    workers=100,
    bs=256
)

# Greedy score: -4.225 (about 4x worse than optimal)


@torch.no_grad()
def generate_curriculum_episode(model: PPOModel, agent: HeuristicAgent):
    env = AStarEnv(target_n=GRAPH_SIZE)
    s = env.state()
    halt = False
    buffer = PPOMemory(bs=1)

    while not halt:
        a = agent.get_action(env)

        a = torch.tensor([[a[0]],[a[1]]])
        _, p, v = model.take_action(s, a)
        src = s.src[a[0]].item()
        dst = s.dst[a[1]].item() - s.offset

        next_s, r, halt = env.step(src,dst)
        buffer.remember(s, a, v, p, r, halt)
        s = next_s

    return buffer

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
    def get_episodes(i, e, model):
        if i % 10 == 9 and e < 25:
            return generate_curriculum_episode(model, GreedyAgent())
        else:
            return generate_episode(model)

    log = []
    for e in range(hp.epochs):
        mems = Parallel(n_jobs=hp.workers, prefer='processes')(
            delayed(get_episodes)(i, e, model)
            for i in range(hp.eps_per_update)
        )
        avg_r = [sum(mem.r) for mem in mems]
        avg_r = sum(avg_r) / len(avg_r)

        buff = PPOMemory(hp.bs).load(mems)
        model.memory = buff
        model.learn()

        print(f'[{e+1}] Average reward: {avg_r}')
        log.append(avg_r)

        torch.save(log, 'logs/node_mapper.pt')

model = PPOModel(in_dim=5, hidden=256, lr=0.001, epochs=1)
train(model, HP)