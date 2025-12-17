from typing import List, Optional
from threading import Lock

class GlobalBest:
    def __init__(self):
        self.route: Optional[List[int]] = None
        self.cost: float = float("inf")
        self.agent_id: Optional[str] = None
        self._lock = Lock()

    def try_update(
        self,
        candidate_route: List[int],
        candidate_cost: float,
        agent_id: str
    ) -> bool:
        """
        Tenta atualizar o g_best de forma atômica.
        Retorna True se atualizou.
        """
        with self._lock:
            if candidate_cost < self.cost:
                self.route = candidate_route.copy()
                self.cost = candidate_cost
                self.agent_id = agent_id
                return True
        return False

    def get(self):
        """
        Retorna uma cópia do g_best atual.
        """
        with self._lock:
            return self.route, self.cost, self.agent_id


# Instância única (singleton)
global_best = GlobalBest()
