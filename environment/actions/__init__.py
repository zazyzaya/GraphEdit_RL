from .simple_actions import Action, AddNode, DeleteNode, ChangeFeature, AcceptingEdges, NotAcceptingEdges, AddEdge
from .graphlet_actions import N1, N2, N3, N4


SIMPLE_ACTIONS = [
    AddNode,
    ChangeFeature,
    DeleteNode,
    AcceptingEdges,
    NotAcceptingEdges,
    AddEdge
]
NEEDS_COLOR = lambda x : False if x in [2,3,4,5] else True

ACTION_TO_IDX = dict()
for act in SIMPLE_ACTIONS:
    ACTION_TO_IDX[act] = len(ACTION_TO_IDX)
for act in N1:
    ACTION_TO_IDX[act] = len(ACTION_TO_IDX)
for act in N2:
    ACTION_TO_IDX[act] = len(ACTION_TO_IDX)
for act in N3:
    ACTION_TO_IDX[act] = len(ACTION_TO_IDX)
for act in N4:
    ACTION_TO_IDX[act] = len(ACTION_TO_IDX)

N_ACTIONS = len(ACTION_TO_IDX)
IDX_TO_ACTION = {v:k for k,v in ACTION_TO_IDX.items()}