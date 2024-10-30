from abc import ABC, abstractmethod
from copy import deepcopy
from environment.env import AStarEnv

class HeuristicAgent(ABC):
    @abstractmethod
    def get_action(self, env:AStarEnv):
        pass


class GreedyAgent(HeuristicAgent):
    def get_action(self, env: AStarEnv):
        possible_src = list(set(range(env.start.n)) - set(env.mapping.keys()))
        possible_dst = list(set(range(env.end.n)) - set(env.inv_mapping.keys()))

        possible_src.sort()
        possible_dst.sort()

        best = [float('-inf'), None]
        for i,src in enumerate(possible_src):
            for j,dst in enumerate(possible_dst):
                _,r,_ = deepcopy(env).step(src,dst)

                if r > best[0]:
                    best = [r, (i,j)]

        return best[1]