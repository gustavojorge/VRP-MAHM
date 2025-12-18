import random
from typing import Dict, Optional

from src.agent_beliefs import AgentBeliefs


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
    strategy: str = "roulette",
    epsilon_exploration: float = 0.15,
    logger: Optional[object] = None
) -> str:
    """
    Decide which metaheuristic the agent should execute.

    strategy:
        - "roulette" : proportional roulette with epsilon-greedy exploration
        - "greedy"   : best action at current state
        - "random"   : random choice
    
    epsilon_exploration: probability of random exploration (0.0 = pure exploitation, 1.0 = pure exploration)
                         Default: 0.15 (15% exploration, 85% exploitation)
    """

    if strategy == "random":
        return random.choice(list(beliefs.actions.keys()))

    if strategy == "greedy":
        return beliefs.get_best_action()

    if strategy == "roulette":
        # Epsilon-greedy: randomly select an action with probability epsilon_exploration
        if random.random() < epsilon_exploration:
            return random.choice(list(beliefs.actions.keys()))
        
        # Exploitation: use scores with minimum epsilon value for exploration
        scores = beliefs.get_all_action_scores(epsilon=1.0)
        if logger:
            logger.log(f"Scores: {scores}")
        return roulette_wheel_selection(scores)

    raise ValueError(f"[ERROR] Unknown decision strategy: {strategy}")
