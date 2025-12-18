import random
from typing import Dict

from agents.agent_beliefs import AgentBeliefs


def roulette_wheel_selection(scores: Dict[str, float]) -> str:
    """
    Selection by proportional roulette based on scores.
    """
    total_score = sum(scores.values())

    # If all scores are zero we select a random action
    if total_score == 0:
        return random.choice(list(scores.keys()))

    r = random.uniform(0, total_score)
    cumulative = 0.0

    for action, score in scores.items():
        cumulative += score
        if r <= cumulative:
            return action

    # If no action is selected we select a random action
    return random.choice(list(scores.keys()))


def decision_method(
    beliefs: AgentBeliefs,
    strategy: str = "roulette"
) -> str:
    """
    Decide which metaheuristic the agent should execute.

    strategy:
        - "roulette" : proportional roulette
        - "greedy"   : best action at current state
        - "random"   : random choice
    """

    if strategy == "random":
        return random.choice(list(beliefs.actions.keys()))

    if strategy == "greedy":
        return beliefs.get_best_action()

    if strategy == "roulette":
        scores = beliefs.get_all_action_scores()
        print("Scores: ", scores)
        return roulette_wheel_selection(scores)

    raise ValueError(f"[ERROR] Unknown decision strategy: {strategy}")
