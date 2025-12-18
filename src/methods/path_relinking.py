from typing import List, Callable, Tuple

from src.utils.evaluator import evaluate_route


def path_relinking(
    origin: List[int],
    target: List[int],
    instance: dict,
    intensification_method: Callable[[List[int], dict], Tuple[List[int], float]]
) -> Tuple[List[int], float]:
    """
    Path-Relinking para problemas de permutação
    com Intensificação Oportunista e verificação de viabilidade.

    origin  : Ps — posição atual do agente
    target  : Pt — g_best ou solução elite
    """

    # Cópia defensiva
    current = origin.copy()

    # Avaliação da origem (baseline)
    origin_feasible, origin_cost = evaluate_route(origin, instance)
    if not origin_feasible:
        raise ValueError("Path-Relinking iniciado com solução inviável")

    # Melhor solução ao longo do caminho
    best_route = origin.copy()
    best_cost = origin_cost

    # Ignora o depósito (posição 0 e última)
    positions = range(1, len(origin) - 1)

    for i in positions:

        # Se já está igual ao target, não faz nada
        if current[i] == target[i]:
            continue

        # Encontra a posição do nó desejado
        j = current.index(target[i])

        # Swap direcionado
        current[i], current[j] = current[j], current[i]

        feasible, current_cost = evaluate_route(current, instance)

        # ❌ Solução inviável → ignora e continua
        if not feasible:
            continue

        # ✔ Atualiza melhor solução do caminho
        if current_cost < best_cost:
            best_route = current.copy()
            best_cost = current_cost

        # 🔴 Parada oportunista (melhor que Ps)
        if current_cost < origin_cost:
            intensified_route, intensified_cost = intensification_method(
                current.copy(), instance
            )

            # Blindagem final
            feasible_int, cost_int = evaluate_route(intensified_route, instance)
            if feasible_int:
                return intensified_route, intensified_cost
            else:
                # Se a intensificação falhar, mantém melhor do caminho
                return best_route, best_cost

    # 🟦 Nenhuma melhoria relevante encontrada
    return best_route, best_cost
