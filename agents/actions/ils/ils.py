from typing import List

from agents.utils.compute_route_cost import compute_route_cost
from agents.actions.vnd.vnd import vnd
from agents.actions.ils.perturbation import perturb_route

def ils(
    initial_route: List[int],
    trip_time_matrix: List[List[int]],
    max_iterations: int = 50
) -> tuple[List[int], int]:
    """
    Iterated Local Search (ILS)
    """

    # Intensificação inicial
    current_route, current_cost = vnd(initial_route, trip_time_matrix)

    best_route = current_route
    best_cost = current_cost

    for _ in range(max_iterations):
        # Diversificação
        perturbed = perturb_route(current_route, k=2)

        # Intensificação
        new_route, new_cost = vnd(perturbed, trip_time_matrix)

        # Aceitação (greedy)
        if new_cost < best_cost:
            best_route = new_route
            best_cost = new_cost
            current_route = new_route
            current_cost = new_cost

    return best_route, best_cost
