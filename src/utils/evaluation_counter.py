"""
Evaluation counter module for tracking objective function evaluations.

This module provides thread-local context management to identify which agent
is making evaluation calls, allowing proper per-agent evaluation counting.
"""

import threading
from typing import Dict, Optional

# Thread-local storage for agent context
_local_context = threading.local()


def set_agent_context(agent_id: str) -> None:
    """
    Set the current agent context for evaluation counting.
    
    Args:
        agent_id: The ID of the agent making evaluations
    """
    _local_context.agent_id = agent_id


def clear_agent_context() -> None:
    """
    Clear the current agent context.
    """
    if hasattr(_local_context, 'agent_id'):
        delattr(_local_context, 'agent_id')


def get_agent_context() -> Optional[str]:
    """
    Get the current agent context.
    
    Returns:
        The current agent ID, or None if not set
    """
    return getattr(_local_context, 'agent_id', None)


def increment_evaluation(agent_counters: Dict[str, int]) -> None:
    """
    Increment the evaluation counter for the current agent.
    
    This function should be called from compute_route_cost to count
    each objective function evaluation.
    
    Args:
        agent_counters: Shared dictionary mapping agent_id to evaluation count
    """
    agent_id = get_agent_context()
    if agent_id is not None:
        # Initialize counter if it doesn't exist
        if agent_id not in agent_counters:
            agent_counters[agent_id] = 0
        # Increment the counter
        agent_counters[agent_id] = agent_counters[agent_id] + 1


def get_agent_evaluation_count(agent_id: str, agent_counters: Dict[str, int]) -> int:
    """
    Get the evaluation count for a specific agent.
    
    Args:
        agent_id: The agent ID
        agent_counters: Shared dictionary mapping agent_id to evaluation count
    
    Returns:
        The evaluation count for the agent, or 0 if not found
    """
    return agent_counters.get(agent_id, 0)

