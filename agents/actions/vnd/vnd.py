from typing import List, Callable

from agents.utils.compute_route_cost import compute_route_cost
from agents.actions.vnd.neighborhoods.swap import swap_neighborhood
from agents.actions.vnd.neighborhoods.two_opt import two_opt_neighborhood

def vnd(
    initial_route: List[int],
    trip_time_matrix: List[List[int]]
) -> tuple[List[int], int]:
    """
    Variable Neighborhood Descent (VND)
    """

    neighborhoods: List[Callable[[List[int]], List[List[int]]]] = [
        swap_neighborhood,
        two_opt_neighborhood
    ]

    current_route = initial_route
    current_cost = compute_route_cost(current_route, trip_time_matrix)

    k = 0
    while k < len(neighborhoods):
        neighborhood = neighborhoods[k]
        best_route = current_route
        best_cost = current_cost

        for neighbor in neighborhood(current_route):
            cost = compute_route_cost(neighbor, trip_time_matrix)

            if cost < best_cost:
                best_route = neighbor
                best_cost = cost

        if best_cost < current_cost:
            current_route = best_route
            current_cost = best_cost
            k = 0  # volta para a primeira vizinhança
        else:
            k += 1  # próxima vizinhança

    return current_route, current_cost
