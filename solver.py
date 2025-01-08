from copy import deepcopy

import torch
from tqdm import tqdm
import pandas as pd

from environment.env import AStarEnv
from models.node_mapper import load_model

class Node():
    def __init__(self, env: AStarEnv, src, dst, g=0,h=0, terminal=False):
        self.src = src; self.dst = dst
        self.env = env
        self.h = h
        self.g = g
        self.f = g + h
        self.terminal = terminal

        self.children = []

    def explore(self, heuristic):
        if self.terminal:
            return []

        avail_src = set(range(self.env.start.n)) - set(self.env.mapping.keys())
        avail_dst = set(range(self.env.end.n)) - set(self.env.inv_mapping.keys())

        for src in avail_src:
            for dst in avail_dst:
                env = deepcopy(self.env)
                _, r, terminal = env.step(src,dst, return_state=False)

                if not terminal:
                    h_value = heuristic(env)
                else:
                    h_value = 0

                self.children.append(
                    Node(env, src, dst, g=r+self.g, h=h_value, terminal=terminal)
                )

        # No longer needed after exploration
        del self.env
        return self.children


class AStarSearch():
    def __init__(self,env):
        self.env = env

    def search(self, heuristic):
        root = Node(self.env, None, None, 0, 0, False)
        steps = 0

        domain = root.explore(heuristic)
        while True:
            domain.sort(key=lambda x : x.f)
            child = domain.pop()

            if child.terminal:
                break

            domain = child.explore(heuristic) + domain
            steps += 1

        return child, steps

model = load_model('model.pt')
model.eval()

@torch.no_grad()
def ppo_heuristic(env):
    _, value = model.forward(env.state())
    return value.item()

def greedy(env):
    return 0

'''
N = 3  (runtime ~2m)
greedy_steps    173.41
ppo_steps       169.95
ppo_correct       1.00
dtype: float64

N = 4 (runtime ~2hrs)
greedy_steps    4457.38
ppo_steps       4379.09
ppo_correct        1.00
dtype: float64
Ratio: 0.9824358703992031
'''
if __name__ == '__main__':
    stats = {
        'greedy_steps': [],
        'ppo_steps': [],
        'ppo_correct': []
    }
    for _ in tqdm(range(100)):
        env = AStarEnv(4, scrambles=10)

        search = AStarSearch(env)
        c_g,   steps_g = search.search(greedy)
        c_rl, steps_rl = search.search(ppo_heuristic)

        stats['greedy_steps'].append(steps_g)
        stats['ppo_steps'].append(steps_rl)
        stats['ppo_correct'].append(1 if c_g.g >= c_rl.g else 0)

    df = pd.DataFrame(stats)
    mean = df.mean()
    print(mean)
    print(f"Ratio: {mean['ppo_steps'] / mean['greedy_steps']}")
