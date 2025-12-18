from typing import List, Tuple, Callable
import random

from agents.problem.evaluator import evaluate_route
from agents.actions.vnd.vnd import vnd
from agents.actions.vnd.neighborhoods.swap import swap_neighborhood
from agents.actions.vnd.neighborhoods.two_opt import two_opt_neighborhood


def vns(
    initial_route: List[int],
    instance: dict,
    max_iterations: int = 50,
    seed: int | None = None,
    max_shake_tries: int = 10,
) -> Tuple[List[int], float]:
    """
    Variable Neighborhood Search (VNS) básico

    Estratégia:
    - Para cada iteração, percorre-se a lista de vizinhanças (shaking)
    - Para cada vizinhança aplica-se um movimento aleatório (shaking)
    - Aplica-se uma intensificação com VND a partir da solução perturbada
    - Se houver melhoria a solução corrente é atualizada e reiniciam-se as vizinhanças

    Parâmetros:
    - max_iterations: número máximo de iterações (cada iteração tenta todas as vizinhanças)
    - seed: semente opcional para reprodutibilidade
    - max_shake_tries: número de tentativas para gerar um vizinho viável na fase de shaking
    """

    if seed is not None:
        random.seed(seed)

    neighborhoods: List[Callable[[List[int]], List[List[int]]]] = [
        swap_neighborhood,
        two_opt_neighborhood,
    ]

    # Avaliação inicial
    feasible, current_cost = evaluate_route(initial_route, instance)
    if not feasible:
        raise ValueError("VNS recebeu uma rota inicial inviável")

    current_route = initial_route.copy()

    best_route = current_route
    best_cost = current_cost

    for it in range(max_iterations):
        # percorre vizinhanças
        k = 0
        while k < len(neighborhoods):
            neighborhood = neighborhoods[k]

            # shaking: tenta gerar um vizinho viável aleatório
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
                # não foi possível gerar um vizinho viável nesta vizinhança
                k += 1
                continue

            # intensificação com VND
            new_route, new_cost = vnd(shaken_route, instance)

            # aceitação se melhorar
            if new_cost < current_cost:
                current_route = new_route
                current_cost = new_cost

                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost

                # reinicia vizinhanças
                k = 0
            else:
                k += 1

    return best_route, best_cost
