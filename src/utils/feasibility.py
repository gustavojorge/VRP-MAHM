def is_feasible_route(route: list[int], instance: dict) -> bool:
    """
    Checks if the route respects:
    - starts and ends at the depot
    - visits each node only once
    - dynamic capacity constraint
    """

    nodes = instance["nodes"]
    max_capacity = instance["vehicle_fleet"]["max_capacity"]

    # 1. Starts and ends at the depot
    if route[0] != 0 or route[-1] != 0:
        return False

    # 2. Visits each active node only once
    active_nodes = {n["id"] for n in nodes if n["id"] != 0}
    route_nodes = route[1:-1]

    if set(route_nodes) != active_nodes:
        return False

    # 3. Capacity constraint
    load = 0
    node_map = {n["id"]: n for n in nodes}

    for v in route_nodes:
        node = node_map[v]
        load += node["n_boardings"] - node["n_alighting"]

        if load < 0 or load > max_capacity:
            return False

    return True
