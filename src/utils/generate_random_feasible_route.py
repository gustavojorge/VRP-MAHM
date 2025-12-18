import random

def generate_random_feasible_route(instance: dict) -> list[int]:
    nodes = instance["nodes"]
    trip_time = instance["trip_time_matrix"]
    max_capacity = instance["vehicle_fleet"]["max_capacity"]

    # Nós ativos (exclui o depósito 0)
    active_nodes = [n["id"] for n in nodes if n["id"] != 0]

    route = [0]  # começa no depósito
    pending = active_nodes.copy()
    current_load = 0

    while pending:
        candidates = []

        for v in pending:
            node = next(n for n in nodes if n["id"] == v)
            new_load = current_load + node["n_boardings"] - node["n_alighting"]

            if 0 <= new_load <= max_capacity:
                candidates.append((v, new_load))

        if not candidates:
            raise RuntimeError("Não foi possível gerar rota viável (Vcandidato vazio).")

        chosen, new_load = random.choice(candidates)
        route.append(chosen)
        pending.remove(chosen)
        current_load = new_load

    route.append(0)  # fecha o ciclo
    return route

