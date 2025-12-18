from typing import List, Tuple

from src.actions.vnd.vnd import vnd
from src.actions.ils.perturbation import perturb_route
from src.utils.evaluator import evaluate_route


def ils(
    initial_route: List[int],
    instance: dict,
    max_iterations: int = 50
) -> Tuple[List[int], float]:
    """
    Iterated Local Search (ILS)
    """

    current_route, current_cost = vnd(initial_route, instance)

    best_route = current_route
    best_cost = current_cost

    for it in range(max_iterations):


        perturbed_route = perturb_route(current_route, k=2)

        feasible, _ = evaluate_route(perturbed_route, instance)
        if not feasible:
            continue

        new_route, new_cost = vnd(perturbed_route, instance)

        if new_cost < best_cost:
            best_route = new_route
            best_cost = new_cost
            current_route = new_route
            current_cost = new_cost
        else:
            pass

    return best_route, best_cost
