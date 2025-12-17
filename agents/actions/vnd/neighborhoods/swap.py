from typing import List

def swap_neighborhood(route: List[int]) -> List[List[int]]:
    neighbors = []
    n = len(route)

    for i in range(1, n - 1):
        for j in range(i + 1, n - 1):
            new_route = route.copy()
            new_route[i], new_route[j] = new_route[j], new_route[i]
            neighbors.append(new_route)

    return neighbors
