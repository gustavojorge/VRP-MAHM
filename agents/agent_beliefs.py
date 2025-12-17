from typing import Dict, List, Optional


class ActionStats:
    """
    Estatísticas associadas a uma metaheurística (ação).
    """

    def __init__(self, name: str):
        self.name = name
        self.times_selected: int = 0
        self.times_success: int = 0
        self.total_improvement: float = 0.0
        self.last_improvement: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.times_selected == 0:
            return 0.0
        return self.times_success / self.times_selected

    @property
    def avg_improvement(self) -> float:
        if self.times_success == 0:
            return 0.0
        return self.total_improvement / self.times_success


class AgentBeliefs:
    """
    Representa as crenças internas do agente sobre:
    - desempenho das metaheurísticas
    - estado atual
    - melhor solução pessoal (p_best)
    """

    def __init__(
        self,
        agent_id: str,
        metaheuristics: List[str]
    ):
        self.agent_id = agent_id

        # Estatísticas das ações (metaheurísticas)
        self.actions: Dict[str, ActionStats] = {
            name: ActionStats(name) for name in metaheuristics
        }

        # Estado atual do agente
        self.current_route: Optional[List[int]] = None
        self.current_cost: float = float("inf")

        # Melhor solução pessoal (p_best)
        self.p_best_route: Optional[List[int]] = None
        self.p_best_cost: float = float("inf")

    # ------------------------------------------------------------------
    # Atualização após execução de uma ação (Learning Method)
    # ------------------------------------------------------------------

    def update_after_action(
        self,
        action_name: str,
        old_cost: float,
        new_cost: float
    ) -> None:
        """
        Atualiza as crenças após a execução de uma metaheurística.
        """
        stats = self.actions[action_name]
        stats.times_selected += 1

        improvement = old_cost - new_cost
        stats.last_improvement = improvement

        if improvement > 0:
            stats.times_success += 1
            stats.total_improvement += improvement

    # ------------------------------------------------------------------
    # Consulta para o Decision Method
    # ------------------------------------------------------------------

    def get_action_stats(self, action_name: str) -> ActionStats:
        return self.actions[action_name]

    def get_all_action_scores(self) -> Dict[str, float]:
        """
        Retorna um score simples para cada ação.
        Pode ser usado diretamente na roleta.
        """
        scores = {}
        for name, stats in self.actions.items():
            # Score simples: sucesso + melhoria média
            scores[name] = stats.times_success + stats.avg_improvement
        return scores

    def get_best_action(self) -> str:
        """
        Retorna a ação com maior score atual.
        Usado para intensificação oportunista.
        """
        scores = self.get_all_action_scores()
        return max(scores, key=scores.get)

    # ------------------------------------------------------------------
    # Estado do agente
    # ------------------------------------------------------------------

    def update_current_solution(
        self,
        route: List[int],
        cost: float
    ) -> None:
        self.current_route = route
        self.current_cost = cost

    def try_update_pbest(
        self,
        route: List[int],
        cost: float
    ) -> bool:
        if cost < self.p_best_cost:
            self.p_best_cost = cost
            self.p_best_route = route.copy()
            return True
        return False
