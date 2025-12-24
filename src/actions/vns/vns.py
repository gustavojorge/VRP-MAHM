from typing import List, Tuple, Callable
import random

from src.utils.evaluator import evaluate_route
from src.actions.vnd.vnd import vnd
from src.actions.vnd.neighborhoods.swap import swap_neighborhood
from src.actions.vnd.neighborhoods.two_opt import two_opt_neighborhood


def vns(
    initial_route: List[int],
    instance: dict,
    max_iterations: int = 50,
    seed: int | None = None,
    max_shake_tries: int = 10,
) -> Tuple[List[int], float]:
    """
    Variable Neighborhood Search (VNS)

    Strategy:
    - For each iteration, traverse the list of neighborhoods (shaking)
    - For each neighborhood, apply a random movement (shaking)
    - Apply an intensification with VND from the perturbed solution
    - If there is improvement, the current solution is updated and the neighborhoods are restarted

    Parameters:
    - max_iterations: maximum number of iterations (each iteration tries all neighborhoods)
    - seed: optional seed for reproducibility
    - max_shake_tries: number of tries to generate a feasible neighbor in the shaking phase
    """

    if seed is not None:
        random.seed(seed)

    neighborhoods: List[Callable[[List[int]], List[List[int]]]] = [
        swap_neighborhood,
        two_opt_neighborhood,
    ]

    # Initial evaluation
    feasible, current_cost = evaluate_route(initial_route, instance)
    if not feasible:
        raise ValueError("VNS received an infeasible initial route")

    current_route = initial_route.copy()

    best_route = current_route
    best_cost = current_cost

    for it in range(max_iterations):
        # traverse neighborhoods
        k = 0
        while k < len(neighborhoods):
            neighborhood = neighborhoods[k]

            # shaking: try to generate a random feasible neighbor
            shaken_route = None
            for _ in range(max_shake_tries):
                neighbors = neighborhood(current_route)
                if not neighbors:
                    break
                candidate = random.choice(neighbors)
                feasible_candidate, _ = evaluate_route(candidate, instance)
                if feasible_candidate:
                    shaken_route = candidate
                    break

            if shaken_route is None:
                # it was not possible to generate a feasible neighbor in this neighborhood
                k += 1
                continue

            # intensification with VND
            new_route, new_cost = vnd(shaken_route, instance)

            # acceptance if improved
            if new_cost < current_cost:
                current_route = new_route
                current_cost = new_cost

                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost

                # restart neighborhoods
                k = 0
            else:
                k += 1

    return best_route, best_cost
