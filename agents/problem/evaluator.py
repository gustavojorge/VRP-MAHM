from agents.problem.feasibility import is_feasible_route
from agents.utils.compute_route_cost import compute_route_cost

def evaluate_route(route: list[int], instance: dict) -> float:
    """
    Retorna o custo se viável, senão infinito.
    """
    if not is_feasible_route(route, instance):
        return float("inf")

    return compute_route_cost(route, instance["trip_time_matrix"])
