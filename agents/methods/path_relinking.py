from typing import List, Callable, Tuple

from agents.problem.evaluator import evaluate_route

def path_relinking(
    origin: List[int],
    target: List[int],
    trip_time_matrix: List[List[int]],
    intensification_method: Callable[[List[int], List[List[int]]], Tuple[List[int], float]]
) -> Tuple[List[int], float]:
    """
    Path-Relinking para problemas de permutação
    com Intensificação Oportunista e verificação de viabilidade.

    origin  : Ps — posição atual do agente
    target  : Pt — g_best ou solução elite
    """

    # Copiamos para não alterar a solução original
    current = origin.copy()

    # Avaliação da origem (baseline)
    origin_feasible, origin_cost = evaluate_route(origin, trip_time_matrix)

    if not origin_feasible:
        raise ValueError("Path-Relinking iniciado com solução inviável")

    # Melhor solução encontrada ao longo do caminho
    best_route = origin.copy()
    best_cost = origin_cost

    # Ignora depósito (posição 0 e última)
    positions = list(range(1, len(origin) - 1))

    for i in positions:

        # Se o elemento já está correto, pula
        if current[i] == target[i]:
            continue

        # Encontra onde está o elemento desejado
        j = current.index(target[i])

        # Swap direcionado (aumenta similaridade com target)
        current[i], current[j] = current[j], current[i]

        is_feasible, current_cost = evaluate_route(current, trip_time_matrix)

        # 🔴 Se inviável, ignora e continua o caminho
        if not is_feasible:
            continue

        # Atualiza melhor do caminho
        if current_cost < best_cost:
            best_route = current.copy()
            best_cost = current_cost

        # 🔴 Parada oportunista (melhor que Ps)
        if current_cost < origin_cost:
            # Intensificação com a meta-heurística escolhida
            return intensification_method(current, trip_time_matrix)

    # 🟦 Nenhuma melhoria relevante encontrada
    return best_route, best_cost
