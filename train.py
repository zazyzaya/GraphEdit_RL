import random
from types import SimpleNamespace

from joblib import Parallel, delayed
import torch
from tqdm import tqdm

from embed_all import get_action_embs
from environment.actions import Action, SIMPLE_ACTIONS, N1, N2, N3, N4
from environment.env import GraphEnv
from models.ppo import GraphPPO, PPOMemory


SEED = 1337
MAX_EPISODE_LEN = 100
GRAPH_SIZE = 3
SIZE_VARIANCE = 0

GET_N = lambda gs, sv : gs + (random.randint(0, sv*2)-sv)

HP = SimpleNamespace(
    epochs=100_000,
    eps_per_update=25,
    workers=25,
    bs = 64,
    hidden = 256
)

def translate_action(a_idx: int) -> Action:
    nid = a_idx // ACT_EMBS.size(0)
    act = a_idx  % ACT_EMBS.size(0)
    act,param =  ACTIONS[act]

    return act(target=nid, ntype=param)

def generate_cheating_episode(agent: GraphPPO, i: int, graph_size, size_var):
    torch.set_num_threads(1)
    n = GET_N(graph_size, size_var)
    env = GraphEnv(
        n,
        n_initial=n,
        p = 0.1 if n > 5 else 0.5
    )

    s = env.reset()
    mem = PPOMemory(0)

    for i in tqdm(range(random.randint(5,50)), desc=str(i)):
        a,v,p = agent.get_action(*s)
        act_obj = translate_action(a)
        next_s, r, t = env.step(act_obj)

        mem.remember(s,a,v,p,r,t)
        s = next_s

        # If we actually solved one for real
        if t and i != MAX_EPISODE_LEN-1:
            return mem

    # Take whatever state the current graph became halfway into the
    # episode, and pretend like that was the target all along
    s0 = mem.s[0][0]
    new_target = mem.s[-1][0]
    actions = mem.a

    # Reload the env
    env.reset()
    env.current = s0
    env.set_target(new_target)

    # Reset buffer
    mem = PPOMemory(0)

    # Play game again, with predetermined actions that will get us to the
    # target graph at least in MAX_EPS / 2 steps (maybe sooner)
    s = env.get_state()
    for a in actions:
        a,v,p = agent.take_action(*s, torch.tensor(a))
        act_obj = translate_action(a)
        next_s, r, t = env.step(act_obj)

        mem.remember(s,a,v,p,r,t)
        s = next_s
        if t:
            break
    else:
        print(s)

    return mem


def generate_episode(agent: GraphPPO, i: int, graph_size, size_var):
    torch.set_num_threads(1)
    n = GET_N(graph_size, size_var)
    env = GraphEnv(
        n,
        n_initial=n,
        p = 0.1 if n > 5 else 0.5
    )

    s = env.reset()
    mem = PPOMemory(0)

    for i in tqdm(range(MAX_EPISODE_LEN), desc=str(i)):
        a,v,p = agent.get_action(*s)
        act_obj = translate_action(a)
        next_s, r, t = env.step(act_obj)

        mem.remember(s,a,v,p,r,t)
        s = next_s

        if t:
            break

    return mem

def train(agent: GraphPPO):
    global GRAPH_SIZE, SIZE_VARIANCE
    log = dict(lens=[], r=[])


    for e in range(1,HP.epochs):
        memories = Parallel(n_jobs=HP.workers, prefer='processes')(
            delayed(generate_episode)(agent, i, GRAPH_SIZE, SIZE_VARIANCE)
            for i in range(HP.eps_per_update)
        )
        lens = [len(m.r) for m in memories]
        avg_len = sum(lens) / HP.eps_per_update
        avg_r = sum([sum(m.r) for m in memories]) / HP.eps_per_update

        if avg_len == MAX_EPISODE_LEN:
            print("Generating some cheating games")
            more_memories = Parallel(n_jobs=HP.workers, prefer='processes')(
                delayed(generate_cheating_episode)(agent, i, GRAPH_SIZE, SIZE_VARIANCE)
                for i in range(HP.eps_per_update)
            )

            # Suppliment memories of real games with memories of
            # games where the agent cheats and we retroactively make
            # whatever graph it got to at ts 50 the target the whole time
            memories += more_memories # type: ignore

        agent.memory = PPOMemory(HP.bs).combine(memories)

        log['r'].append(avg_r)
        log['lens'].append(avg_len)

        torch.save(log, 'log.pt')
        agent.save()

        print(f"[{e}] Avg. reward: {avg_r}, Avg. length: {avg_len}")
        torch.set_num_threads(32)
        agent.learn()

        if e % 1_000 == 0:
            GRAPH_SIZE += 1
            SIZE_VARIANCE = int(GRAPH_SIZE * 0.2)


if __name__ == '__main__':
    TRAIN_ACTIONS = SIMPLE_ACTIONS + N1 + N2
    ACT_EMBS, ACTIONS, ACTION_NAMES = get_action_embs(TRAIN_ACTIONS)

    random.seed(SEED)
    torch.manual_seed(SEED)

    env = GraphEnv(GRAPH_SIZE)
    agent = GraphPPO(env.target.x.size(1), HP.hidden, ACT_EMBS, epochs=1)
    train(agent)