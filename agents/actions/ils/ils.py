from typing import List, Tuple

from agents.actions.vnd.vnd import vnd
from agents.actions.ils.perturbation import perturb_route
from agents.problem.evaluator import evaluate_route


def ils(
    initial_route: List[int],
    instance: dict,
    max_iterations: int = 50
) -> Tuple[List[int], float]:
    """
    Iterated Local Search (ILS)
    """

    # 🔴 Intensificação inicial
    current_route, current_cost = vnd(initial_route, instance)

    best_route = current_route
    best_cost = current_cost

    for it in range(max_iterations):

        print(f"\n--- ILS Iteração {it + 1} ---")

        # 🟦 Diversificação
        perturbed_route = perturb_route(current_route, k=2)
        print("Rota perturbada:", perturbed_route)

        feasible, _ = evaluate_route(perturbed_route, instance)
        if not feasible:
            print("❌ Rota inviável após perturbação — descartada")
            continue

        # 🔴 Intensificação
        new_route, new_cost = vnd(perturbed_route, instance)

        print("Rota após VND:", new_route)
        print("Custo após VND:", new_cost)

        # 🟢 Aceitação greedy
        if new_cost < best_cost:
            print("✔ Nova melhor solução encontrada")
            best_route = new_route
            best_cost = new_cost
            current_route = new_route
            current_cost = new_cost
        else:
            print("➖ Solução não melhorou")

    return best_route, best_cost
