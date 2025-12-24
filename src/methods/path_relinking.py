from typing import List, Callable, Tuple

from src.utils.evaluator import evaluate_route

def path_relinking(
    origin: List[int],
    target: List[int],
    instance: dict,
    intensification_method: Callable[[List[int], dict], Tuple[List[int], float]]
) -> Tuple[List[int], float]:
    """
    Path-Relinking for permutation problems

    origin  : Ps — current position of the agent
    target  : Pt — g_best or elite solution
    """

    # Defensive copy
    current = origin.copy()

    # Evaluation of the origin (baseline)
    origin_feasible, origin_cost = evaluate_route(origin, instance)
    if not origin_feasible:
        raise ValueError("Path-Relinking started with an infeasible solution")

    # Best solution along the path
    best_route = origin.copy()
    best_cost = origin_cost

    # Ignore the depot (position 0 and last)
    positions = range(1, len(origin) - 1)

    for i in positions:

        # If already equal to target, do nothing
        if current[i] == target[i]:
            continue

        # Find the position of the desired node
        j = current.index(target[i])

        # Directed swap
        current[i], current[j] = current[j], current[i]

        feasible, current_cost = evaluate_route(current, instance)

        # Infeasible solution -> ignore and continue
        if not feasible:
            continue

        # Update best solution along the path
        if current_cost < best_cost:
            best_route = current.copy()
            best_cost = current_cost

        # Opportunistic stop (better than Ps)
        if current_cost < origin_cost:
            intensified_route, intensified_cost = intensification_method(
                current.copy(), instance
            )

            feasible_int, cost_int = evaluate_route(intensified_route, instance)
            if feasible_int:
                return intensified_route, intensified_cost
            else:
                # If the intensification fails, keep the best solution along the path
                return best_route, best_cost

    # No relevant improvement found
    return best_route, best_cost
