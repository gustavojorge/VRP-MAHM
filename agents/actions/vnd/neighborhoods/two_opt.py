from typing import List

def two_opt_neighborhood(route: List[int]) -> List[List[int]]:
    neighbors = []
    n = len(route)

    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_route = (
                route[:i] +
                list(reversed(route[i:j + 1])) +
                route[j + 1:]
            )
            neighbors.append(new_route)

    return neighbors
