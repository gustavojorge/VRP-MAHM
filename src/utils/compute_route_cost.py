def compute_route_cost(route: list[int], trip_time_matrix: list[list[int]]) -> int:
    cost = 0
    for i in range(len(route) - 1):
        origin = route[i]
        destination = route[i + 1]
        cost += trip_time_matrix[origin][destination]
    return cost
