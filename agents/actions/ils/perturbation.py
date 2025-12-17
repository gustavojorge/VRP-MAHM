import random
from typing import List


def perturb_route(route: List[int], k: int = 2) -> List[int]:
    """
    Aplica k swaps aleatórios na rota (mantendo depósito fixo)
    """
    new_route = route.copy()
    internal = new_route[1:-1]

    for _ in range(k):
        i, j = random.sample(range(len(internal)), 2)
        internal[i], internal[j] = internal[j], internal[i]

    return [0] + internal + [0]
