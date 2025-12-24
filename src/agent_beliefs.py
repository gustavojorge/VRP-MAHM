from typing import Dict, List, Optional


class ActionStats:
    """
    Statistics associated with a metaheuristic (action).
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
    Represents the internal beliefs of the agent about:
    - performance of the metaheuristics
    - current state/position
    - personal best solution (p_best)
    """

    def __init__(
        self,
        agent_id: str,
        metaheuristics: List[str]
    ):
        self.agent_id = agent_id

        # Statistics of the actions (metaheuristics)
        self.actions: Dict[str, ActionStats] = {
            name: ActionStats(name) for name in metaheuristics
        }

        # Current state of the agent
        self.current_route: Optional[List[int]] = None
        self.current_cost: float = float("inf")

        # Personal best solution (p_best)
        self.p_best_route: Optional[List[int]] = None
        self.p_best_cost: float = float("inf")

    # ------------------------------------------------------------------
    # Update after execution of an action (Learning Method)
    # ------------------------------------------------------------------

    def update_after_action(
        self,
        action_name: str,
        old_cost: float,
        new_cost: float
    ) -> None:
        """
        Update the beliefs after the execution of a metaheuristic.
        """
        if action_name not in self.actions:
            raise ValueError(f"[ERROR] Action '{action_name}' is not registered")
        
        stats = self.actions[action_name]
        stats.times_selected += 1

        improvement = old_cost - new_cost
        stats.last_improvement = improvement

        if improvement > 0:
            stats.times_success += 1
            stats.total_improvement += improvement

    # ------------------------------------------------------------------
    # Queries for the Decision Method
    # ------------------------------------------------------------------

    def get_action_stats(self, action_name: str) -> ActionStats:
        if action_name not in self.actions:
            raise ValueError(f"[ERROR] Action '{action_name}' is not registered")
        return self.actions[action_name]

    def get_all_action_scores(self, epsilon: float = 1.0) -> Dict[str, float]:
        """
        Returns a score for each action with minimum epsilon value for exploration.
        The score balances success count, success rate, and average improvement (normalized).
        
        The score is dynamic and changes even when there's no improvement, because
        the success rate decreases with more unsuccessful attempts.
        
        epsilon: minimum score value added to all actions to ensure exploration
        """
        scores = {}
        for name, stats in self.actions.items():
            # Success rate scaled to 0-10 range (primary factor for dynamic scoring)
            success_rate_score = stats.success_rate * 10.0
            
            # Success count (absolute number of successes)
            success_count = stats.times_success
            
            # Normalized average improvement (divided by 100 to balance)
            normalized_improvement = stats.avg_improvement / 100.0
            
            # Combined score: success rate (primary) + success count + normalized improvement
            base_score = success_rate_score + success_count + normalized_improvement
            
            # Add epsilon to ensure all actions have a chance (exploration)
            scores[name] = base_score + epsilon
        return scores

    def get_best_action(self) -> str:
        """
        Returns the action with the highest current score.
        """
        scores = self.get_all_action_scores()
        if not scores:
            raise ValueError("[ERROR] No actions available to select")
        return max(scores, key=scores.get)

    # ------------------------------------------------------------------
    # State of the agent
    # ------------------------------------------------------------------

    def update_current_solution(
        self,
        route: List[int],
        cost: float
    ) -> None:
        self.current_route = route.copy()
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
