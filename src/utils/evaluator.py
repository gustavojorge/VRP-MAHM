from src.utils.feasibility import is_feasible_route
from src.utils.compute_route_cost import compute_route_cost


def evaluate_route(route: list[int], instance: dict) -> tuple[bool, float]:
    """
    Returns (feasible, cost).
    If infeasible, cost = infinity.
    """
    feasible = is_feasible_route(route, instance)

    if not feasible:
        return False, float("inf")

    cost = compute_route_cost(route, instance["trip_time_matrix"])
    return True, cost
