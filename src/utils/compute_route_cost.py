# Module-level variable to store the shared agent_counters dictionary
# This is set by each agent process and accessed during evaluation counting
_agent_counters: dict = None


def set_agent_counters(agent_counters: dict) -> None:
    """
    Set the shared agent_counters dictionary for this process.
    Called by each agent process to provide access to the shared counter.
    
    Args:
        agent_counters: Shared dictionary mapping agent_id to evaluation count
    """
    global _agent_counters
    _agent_counters = agent_counters


def compute_route_cost(route: list[int], trip_time_matrix: list[list[int]]) -> int:
    """
    Compute the cost of a route and count this as an objective function evaluation.
    
    Args:
        route: List of node IDs representing the route
        trip_time_matrix: Matrix of travel times between nodes
    
    Returns:
        Total cost of the route
    """
    # Increment evaluation counter if agent_counters is available
    if _agent_counters is not None:
        from src.utils.evaluation_counter import increment_evaluation
        increment_evaluation(_agent_counters)
    
    cost = 0
    for i in range(len(route) - 1):
        origin = route[i]
        destination = route[i + 1]
        cost += trip_time_matrix[origin][destination]
    return cost
