from types import SimpleNamespace
from random import random, randint

from joblib import Parallel, delayed
import torch

from environment.env import AStarEnv
from environment.heuristic_agents import HeuristicAgent, GreedyAgent
from models.node_mapper import PPOModel, PPOMemory

torch.set_num_threads(16)

GRAPH_SIZE = 30
CURRICULUM_DURATION = 1000
HP = SimpleNamespace(
    epochs=10_000,
    eps_per_update=100,
    workers=100,
    bs=256
)

# Greedy score: -4.225 (about 4x worse than optimal)

@torch.no_grad()
def generate_perfect_episode(model: PPOModel):
    '''
    Can kind of cheat, since we know original mappings
    are i -> i. Of course, there may be even more optimal
    mappings, but this is a good enough baseline to get
    pretty good scores.
    '''
    env = AStarEnv(target_n=GRAPH_SIZE)
    s = env.state()
    halt = False
    buffer = PPOMemory(bs=1)

    while not halt:
        a = torch.tensor([[0],[0]])
        _, p, v = model.take_action(s, a)
        src = s.src[a[0]].item()
        dst = s.dst[a[1]].item() - s.offset

        next_s, r, halt = env.step(src,dst)
        buffer.remember(s, a, v, p, r, halt)
        s = next_s

    return buffer

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
    env = AStarEnv(target_n=GRAPH_SIZE, seed=randint(0,9))
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
        '''
        Slowly increase probability of playing a real game
        but takes until epoch 500 to have half real, half perfect
        games.
        '''
        #if e < CURRICULUM_DURATION:
        #    if random() < (CURRICULUM_DURATION-e) / CURRICULUM_DURATION:
        #        return generate_perfect_episode(model)
        return generate_episode(model)

    log = []
    for e in range(hp.epochs):
        mems = Parallel(n_jobs=hp.workers, prefer='processes')(
            delayed(get_episodes)(i, e, model)
            for i in range(hp.eps_per_update)
        )
        avg_r = [sum(mem.r) for mem in mems]
        avg_r = sum(avg_r) / len(avg_r)

        print(f'[{e+1}] Average reward: {avg_r}')
        log.append(avg_r)

        buff = PPOMemory(hp.bs).load(mems)
        model.memory = buff
        model.learn()

        torch.save(log, 'logs/node_mapper.pt')
        model.save('model.pt')

# torch.autograd.set_detect_anomaly(True)
model = PPOModel(in_dim=5, hidden=64, lr=0.0001, epochs=1)
train(model, HP)