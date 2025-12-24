from typing import List, Optional
import copy
from threading import Lock


class GlobalBest:
    """
    Class to manage the global best (g_best) shared between agents.
    
    Supports two modes:
    1. Single-threaded: uses Lock() default of threading
    2. Multi-process: receives a manager from multiprocessing
    """
    
    def __init__(self, manager=None):
        """
        Initializes GlobalBest.
        
        Args:
            manager: If provided (multiprocessing.Manager), uses shared structures.
                     If None, uses thread-safe structures for single-process.
        """
        if manager is not None:
            # Multi-process mode
            self._data = manager.dict({
                "route": None,
                "cost": float("inf"),
                "agent_id": None
            })
            self._lock = manager.Lock()
        else:
            # Single-process mode (thread-safe)
            self._data = {
                "route": None,
                "cost": float("inf"),
                "agent_id": None
            }
            self._lock = Lock()

    def try_update(self, candidate_route, candidate_cost, agent_id):
        """Tries to update the g_best if the candidate is better"""
        with self._lock:
            if candidate_cost < self._data["cost"]:
                self._data["route"] = copy.deepcopy(candidate_route)
                self._data["cost"] = candidate_cost
                self._data["agent_id"] = agent_id
                return True
        return False

    def get(self):
        """Returns a safe copy of the current g_best"""
        with self._lock:
            route = self._data["route"]
            return (
                copy.deepcopy(route) if route else None,
                self._data["cost"],
                self._data["agent_id"]
            )

    def reset(self):
        """Resets the g_best"""
        with self._lock:
            self._data["route"] = None
            self._data["cost"] = float("inf")
            self._data["agent_id"] = None


# Singleton for single-agent use
# For multi-agent, it will be replaced by an instance with a manager
global_best = GlobalBest()
