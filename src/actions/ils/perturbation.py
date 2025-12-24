import random
from typing import List

def perturb_route(route: List[int], k: int = 2) -> List[int]:
    """
    Apply k random swaps in the route (keeping the depot fixed)
    """
    new_route = route.copy()
    internal = new_route[1:-1]

    for _ in range(k):
        i, j = random.sample(range(len(internal)), 2)
        internal[i], internal[j] = internal[j], internal[i]

    return [0] + internal + [0]
