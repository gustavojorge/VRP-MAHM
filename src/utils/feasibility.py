def is_feasible_route(route: list[int], instance: dict) -> bool:
    """
    Verifica se a rota respeita:
    - início e fim no depósito
    - visita única a cada nó
    - restrição de capacidade dinâmica
    """

    nodes = instance["nodes"]
    max_capacity = instance["vehicle_fleet"]["max_capacity"]

    # 1. Começa e termina no depósito
    if route[0] != 0 or route[-1] != 0:
        return False

    # 2. Visita única dos nós ativos
    active_nodes = {n["id"] for n in nodes if n["id"] != 0}
    route_nodes = route[1:-1]

    if set(route_nodes) != active_nodes:
        return False

    # 3. Restrição de capacidade
    load = 0
    node_map = {n["id"]: n for n in nodes}

    for v in route_nodes:
        node = node_map[v]
        load += node["n_boardings"] - node["n_alighting"]

        if load < 0 or load > max_capacity:
            return False

    return True
