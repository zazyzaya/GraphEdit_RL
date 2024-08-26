import torch

from environment.actions import ACTION_TO_IDX, NEEDS_COLOR
from environment.generator import N_COLORS, encode_actions
from models.action_encoder import HVAE

def get_action_embs(action_objs, embedder='action_enc.pt'):
    names, actions, constructors, params = [],[],[],[]
    for a in action_objs:
        i = ACTION_TO_IDX[a]

        if NEEDS_COLOR(i):
            for j in range(N_COLORS):
                actions.append(i)
                params.append(j)
                constructors.append((a,j))
                names.append(
                    f'{a.__name__}, {j}'
                )
        else:
            actions.append(i)
            params.append(-1)
            constructors.append((a,-1))
            names.append(
                f'{a.__name__}'
            )

    action_vecs = encode_actions(
        torch.tensor(actions),
        torch.tensor(params)
    )

    args,kwargs,sd = torch.load(embedder, weights_only=True)
    model = HVAE(*args, **kwargs)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        embs,_,_ = model.encode_action(action_vecs)

    return embs, constructors, names
