import random
from typing import Dict

from agents.agent_beliefs import AgentBeliefs


def roulette_wheel_selection(scores: Dict[str, float]) -> str:
    """
    Seleção por roleta proporcional aos scores.
    """
    total_score = sum(scores.values())

    # Caso degenerado: todos os scores são zero
    if total_score == 0:
        return random.choice(list(scores.keys()))

    r = random.uniform(0, total_score)
    cumulative = 0.0

    for action, score in scores.items():
        cumulative += score
        if r <= cumulative:
            return action

    # Fallback de segurança
    return random.choice(list(scores.keys()))


def decision_method(
    beliefs: AgentBeliefs,
    strategy: str = "roulette"
) -> str:
    """
    Decide qual metaheurística o agente deve executar.

    strategy:
        - "roulette" : roleta proporcional (exploração + exploração)
        - "greedy"   : melhor ação atual (intensificação)
        - "random"   : escolha aleatória
    """

    if strategy == "random":
        return random.choice(list(beliefs.actions.keys()))

    if strategy == "greedy":
        return beliefs.get_best_action()

    if strategy == "roulette":
        scores = beliefs.get_all_action_scores()
        return roulette_wheel_selection(scores)

    raise ValueError(f"Estratégia de decisão desconhecida: {strategy}")
