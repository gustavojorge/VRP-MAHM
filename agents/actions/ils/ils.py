from typing import List, Tuple

from agents.actions.vnd.vnd import vnd
from agents.actions.ils.perturbation import perturb_route
from agents.problem.evaluator import evaluate_route

def ils(
    initial_route: List[int],
    trip_time_matrix: List[List[int]],
    max_iterations: int = 50
) -> Tuple[List[int], float]:
    """
    Iterated Local Search (ILS)
    """

    # Intensificação inicial
    current_route, current_cost = vnd(initial_route, trip_time_matrix)

    best_route = current_route
    best_cost = current_cost

    for _ in range(max_iterations):

        # 🟦 Diversificação (não garante viabilidade)
        perturbed_route = perturb_route(current_route, k=2)

        is_feasible, _ = evaluate_route(perturbed_route, trip_time_matrix)
        if not is_feasible:
            continue  # descarta e segue para próxima iteração

        # 🔴 Intensificação (VND já filtra inviáveis)
        new_route, new_cost = vnd(perturbed_route, trip_time_matrix)

        # 🟢 Aceitação (greedy, apenas viáveis)
        if new_cost < best_cost:
            best_route = new_route
            best_cost = new_cost
            current_route = new_route
            current_cost = new_cost

    return best_route, best_cost
