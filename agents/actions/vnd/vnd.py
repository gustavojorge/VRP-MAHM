from typing import List, Callable, Tuple

from agents.problem.evaluator import evaluate_route
from agents.actions.vnd.neighborhoods.swap import swap_neighborhood
from agents.actions.vnd.neighborhoods.two_opt import two_opt_neighborhood

def vnd(
    initial_route: List[int],
    trip_time_matrix: List[List[int]]
) -> Tuple[List[int], int]:
    """
    Variable Neighborhood Descent (VND)
    com verificação explícita de viabilidade
    """

    neighborhoods: List[Callable[[List[int]], List[List[int]]]] = [
        swap_neighborhood,
        two_opt_neighborhood
    ]

    # Avaliação inicial
    current_cost, feasible = evaluate_route(initial_route, trip_time_matrix)
    if not feasible:
        raise ValueError("VND recebeu uma rota inicial inviável")

    current_route = initial_route
    k = 0

    while k < len(neighborhoods):
        neighborhood = neighborhoods[k]

        best_route = current_route
        best_cost = current_cost

        for neighbor in neighborhood(current_route):
            cost, feasible = evaluate_route(neighbor, trip_time_matrix)

            # Filtro de viabilidade
            if not feasible:
                continue

            if cost < best_cost:
                best_route = neighbor
                best_cost = cost

        if best_cost < current_cost:
            current_route = best_route
            current_cost = best_cost
            k = 0  # intensificação: volta à primeira vizinhança
        else:
            k += 1  # diversificação: próxima vizinhança

    return current_route, current_cost
