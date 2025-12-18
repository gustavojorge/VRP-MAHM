from typing import List, Callable, Tuple

from src.utils.evaluator import evaluate_route
from src.actions.vnd.neighborhoods.swap import swap_neighborhood
from src.actions.vnd.neighborhoods.two_opt import two_opt_neighborhood


def vnd(
    initial_route: List[int],
    instance: dict
) -> Tuple[List[int], float]:
    """
    Variable Neighborhood Descent (VND)
    com verificação explícita de viabilidade
    """

    neighborhoods: List[Callable[[List[int]], List[List[int]]]] = [
        swap_neighborhood,
        two_opt_neighborhood
    ]

    # Avaliação inicial
    feasible, current_cost = evaluate_route(initial_route, instance)
    if not feasible:
        raise ValueError("VND recebeu uma rota inicial inviável")

    current_route = initial_route.copy()
    k = 0

    while k < len(neighborhoods):
        neighborhood = neighborhoods[k]

        best_route = current_route
        best_cost = current_cost

        for neighbor in neighborhood(current_route):
            feasible, cost = evaluate_route(neighbor, instance)

            # Filtro de viabilidade
            if not feasible:
                continue

            if cost < best_cost:
                best_route = neighbor
                best_cost = cost

        if best_cost < current_cost:
            current_route = best_route
            current_cost = best_cost
            k = 0
        else:
            k += 1

    return current_route, current_cost
