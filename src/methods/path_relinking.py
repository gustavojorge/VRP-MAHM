from typing import List, Callable, Tuple

from src.utils.evaluator import evaluate_route


def _relink_one_direction(
    origin: List[int],
    target: List[int],
    instance: dict,
    intensification_method: Callable[[List[int], dict], Tuple[List[int], float]],
    origin_cost: float
) -> Tuple[List[int], float]:
    """
    Path-Relinking in one direction using adjacent shifting.
    
    Args:
        origin: Starting solution (Ps)
        target: Target solution (Pt)
        instance: Problem instance
        intensification_method: Method to intensify promising solutions
        origin_cost: Cost of the origin solution (baseline)
    
    Returns:
        Tuple of (best_route, best_cost) or (intensified_route, intensified_cost) if opportunistic stop
    """
    # Defensive copy
    current = origin.copy()
    
    # Evaluation of the origin (baseline)
    origin_feasible, _ = evaluate_route(origin, instance)
    if not origin_feasible:
        raise ValueError("Path-Relinking started with an infeasible solution")
    
    # Best solution along the path
    best_route = origin.copy()
    best_cost = origin_cost
    
    # Ignore the depot (position 0 and last)
    positions = range(1, len(origin) - 1)
    
    for i in positions:
        # If already equal to target, do nothing
        if current[i] == target[i]:
            continue
        
        # Find the position of the desired node
        j = current.index(target[i])
        
        # Adjacent shifting: move element from position j to position i
        # using successive adjacent swaps
        if j > i:
            # Shift left: move j towards i (decreasing positions)
            for pos in range(j, i, -1):
                # Adjacent swap: swap current[pos] with current[pos-1]
                current[pos], current[pos-1] = current[pos-1], current[pos]
                
                # Evaluate after each adjacent shift
                feasible, current_cost = evaluate_route(current, instance)
                
                # Infeasible solution -> ignore and continue shifting
                if not feasible:
                    continue
                
                # Update best solution along the path
                if current_cost < best_cost:
                    best_route = current.copy()
                    best_cost = current_cost
                
                # Opportunistic stop (better than origin)
                if current_cost < origin_cost:
                    intensified_route, intensified_cost = intensification_method(
                        current.copy(), instance
                    )
                    
                    feasible_int, cost_int = evaluate_route(intensified_route, instance)
                    if feasible_int:
                        return intensified_route, intensified_cost
                    else:
                        # If the intensification fails, keep the best solution along the path
                        return best_route, best_cost
        
        elif j < i:
            # Shift right: move j towards i (increasing positions)
            for pos in range(j, i):
                # Adjacent swap: swap current[pos] with current[pos+1]
                current[pos], current[pos+1] = current[pos+1], current[pos]
                
                # Evaluate after each adjacent shift
                feasible, current_cost = evaluate_route(current, instance)
                
                # Infeasible solution -> ignore and continue shifting
                if not feasible:
                    continue
                
                # Update best solution along the path
                if current_cost < best_cost:
                    best_route = current.copy()
                    best_cost = current_cost
                
                # Opportunistic stop (better than origin)
                if current_cost < origin_cost:
                    intensified_route, intensified_cost = intensification_method(
                        current.copy(), instance
                    )
                    
                    feasible_int, cost_int = evaluate_route(intensified_route, instance)
                    if feasible_int:
                        return intensified_route, intensified_cost
                    else:
                        # If the intensification fails, keep the best solution along the path
                        return best_route, best_cost
    
    # No relevant improvement found
    return best_route, best_cost


def path_relinking(
    origin: List[int],
    target: List[int],
    instance: dict,
    intensification_method: Callable[[List[int], dict], Tuple[List[int], float]]
) -> Tuple[List[int], float]:
    """
    Bidirectional Path-Relinking for permutation problems using adjacent shifting.
    
    Executes path-relinking in both directions (origin → target and target → origin)
    and returns the best solution found.
    
    Args:
        origin: Ps — current position of the agent
        target: Pt — g_best or elite solution
        instance: Problem instance
        intensification_method: Method to intensify promising solutions
    
    Returns:
        Tuple of (best_route, best_cost) from the best direction
    """
    # Evaluate origin and target to get baseline costs
    origin_feasible, origin_cost = evaluate_route(origin, instance)
    if not origin_feasible:
        raise ValueError("Path-Relinking started with an infeasible origin solution")
    
    target_feasible, target_cost = evaluate_route(target, instance)
    if not target_feasible:
        raise ValueError("Path-Relinking started with an infeasible target solution")
    
    # Execute path-relinking in forward direction (origin → target)
    result_forward = _relink_one_direction(
        origin=origin,
        target=target,
        instance=instance,
        intensification_method=intensification_method,
        origin_cost=origin_cost
    )
    
    # Execute path-relinking in backward direction (target → origin)
    result_backward = _relink_one_direction(
        origin=target,
        target=origin,
        instance=instance,
        intensification_method=intensification_method,
        origin_cost=target_cost
    )
    
    # Compare results and return the best
    route_forward, cost_forward = result_forward
    route_backward, cost_backward = result_backward
    
    # Return the solution with the lowest cost
    if cost_forward < cost_backward:
        return route_forward, cost_forward
    else:
        return route_backward, cost_backward
