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
MAX_EPISODE_LEN = 20
GRAPH_SIZE = 3
SIZE_VARIANCE = 0

GET_N = lambda gs, sv : gs + (random.randint(0, sv*2)-sv)

HP = SimpleNamespace(
    epochs=100_000,
    eps_per_update=50,
    workers=50,
    bs = 64,
    hidden = 256,
    lr = 1e-5
)

def translate_action(a_idx: int) -> Action:
    nid = a_idx // ACT_EMBS.size(0)
    act = a_idx  % ACT_EMBS.size(0)
    act,param =  ACTIONS[act]

    return act(target=nid, ntype=param)

def generate_cheating_episode(agent: GraphPPO, i: int, graph_size, size_var, ep_len):
    torch.set_num_threads(1)
    n0 = GET_N(graph_size, size_var)
    n1 = GET_N(graph_size, size_var)
    env = GraphEnv(
        n0,
        n_initial=n1,
        p = 0.1 if n0 > 5 else 0.5
    )

    c0,g0 = env.reset()
    s = (c0,g0)
    mem = PPOMemory(0)

    try:
        for i in range(ep_len):
            a,v,p = agent.get_action(*s)
            act_obj = translate_action(a)
            next_s, r, t = env.step(act_obj)

            mem.remember(s,a,v,p,r,t)
            s = next_s

            # If we actually solved one for real
            if t and i != MAX_EPISODE_LEN-1:
                return mem

    except Exception as e:
        # Log what caused the issue
        with open("ErrorLog.txt", 'a') as f:
            f.write(str(e) + '\n')
            f.write("Goal graph:\n")
            f.write(f'\t{g0.x}\n')
            f.write(f'\t{g0.edge_index}\n')
            f.write("Starting graph:\n")
            f.write(f'{c0.x}\n')
            f.write(f'{c0.edge_index}\n')

            for a in mem.a:
                act_obj = translate_action(a)
                f.write(f'{act_obj.__class__.__name__}({act_obj.target})\n')

            f.write(f"\nModel weights: {agent.state_dict()}")
            f.write("\n\n")

        # Just return empty buffer
        return PPOMemory(0)

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
    try:
        s = env.get_state()
        for a in actions:
            a,v,p = agent.take_action(*s, torch.tensor(a))
            act_obj = translate_action(a)
            next_s, r, t = env.step(act_obj)

            mem.remember(s,a,v,p,r,t)
            s = next_s
            if t:
                break

    except Exception as e:
        # Log what caused the issue
        with open("ErrorLog.txt", 'a') as f:
            f.write(str(e) + '\n')
            f.write(f'{mem.s[0][0].x}\n')
            f.write(f'{mem.s[0][0].edge_index}\n')

            for a in mem.a:
                act_obj = translate_action(a)
                f.write(f'{act_obj.__class__.__name__}({act_obj.target})\n')

            f.write("\n\n")

        # Just return empty buffer
        return PPOMemory(0)

    return mem


def generate_episode(agent: GraphPPO, i: int, graph_size, size_var, ep_len):
    torch.set_num_threads(1)
    n = GET_N(graph_size, size_var)
    env = GraphEnv(
        n,
        n_initial=n,
        p = 0.1 if n > 5 else 0.5
    )

    s = env.reset()
    mem = PPOMemory(0)

    for i in range(ep_len):
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

    def job_dispatcher(agent, i, gs, sv, ep_len):
        if i%2:
            return generate_cheating_episode(agent, i, GRAPH_SIZE, SIZE_VARIANCE, ep_len)
        else:
            return generate_episode(agent, i, GRAPH_SIZE, SIZE_VARIANCE, ep_len)

    for e in range(1,HP.epochs):
        if e < 50_000:
            ep_len = 10 + (e // 5_000)

        memories = Parallel(n_jobs=HP.workers, prefer='processes')(
            delayed(job_dispatcher)(agent, i, GRAPH_SIZE, SIZE_VARIANCE, ep_len)
            for i in range(HP.eps_per_update)
        )

        lens = [len(m.r) for m in memories]
        avg_len = sum(lens) / HP.eps_per_update
        avg_r = sum([sum(m.r) for m in memories]) / HP.eps_per_update

        agent.memory = PPOMemory(HP.bs).combine(memories)

        log['r'].append(avg_r)
        log['lens'].append(avg_len)

        torch.save(log, 'log.pt')
        agent.save()

        print(f"[{e}] Avg. reward: {avg_r}, Avg. length: {avg_len},", end='')
        torch.set_num_threads(32)
        loss = agent.learn(verbose=False)
        print(f' Loss: {loss}')

        if e % 5_000 == 0:
            GRAPH_SIZE += 1
            SIZE_VARIANCE = int(GRAPH_SIZE * 0.2)


if __name__ == '__main__':
    TRAIN_ACTIONS = SIMPLE_ACTIONS + N1 + N2
    ACT_EMBS, ACTIONS, ACTION_NAMES = get_action_embs(TRAIN_ACTIONS)

    random.seed(SEED)
    torch.manual_seed(SEED)

    env = GraphEnv(GRAPH_SIZE)
    agent = GraphPPO(env.target.x.size(1), HP.hidden, ACT_EMBS, epochs=1, lr=HP.lr)
    train(agent)