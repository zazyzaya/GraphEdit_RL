from .simple_actions import Action, AddNode, DeleteEdge, DeleteNode, AddEdge, ChangeFeature
from .graphlet_actions import N1, N2, N3, N4

SIMPLE_ACTIONS = [AddNode, AddEdge, DeleteEdge, ChangeFeature]
SIMPLE_ACTION_MAP = dict(
    NODE_LEVEL = [AddNode],
    EDGE_LEVEL = [AddEdge, DeleteEdge],
    FEAT_LEVEL = [ChangeFeature]
)