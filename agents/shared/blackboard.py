from typing import List, Optional
import copy
from threading import Lock


class GlobalBest:
    """
    Classe para gerenciar o melhor global (g_best) compartilhado entre agentes.
    
    Suporta dois modos:
    1. Single-threaded: usa Lock() padrão do threading
    2. Multi-process: recebe um manager do multiprocessing
    """
    
    def __init__(self, manager=None):
        """
        Inicializa GlobalBest.
        
        Args:
            manager: Se fornecido (multiprocessing.Manager), usa estruturas compartilhadas.
                     Se None, usa estruturas thread-safe para single-process.
        """
        if manager is not None:
            # Modo multiprocessing
            self._data = manager.dict({
                "route": None,
                "cost": float("inf"),
                "agent_id": None
            })
            self._lock = manager.Lock()
        else:
            # Modo single-process (thread-safe)
            self._data = {
                "route": None,
                "cost": float("inf"),
                "agent_id": None
            }
            self._lock = Lock()

    def try_update(self, candidate_route, candidate_cost, agent_id):
        """Tenta atualizar o g_best se o candidato for melhor"""
        with self._lock:
            if candidate_cost < self._data["cost"]:
                self._data["route"] = copy.deepcopy(candidate_route)
                self._data["cost"] = candidate_cost
                self._data["agent_id"] = agent_id
                return True
        return False

    def get(self):
        """Retorna uma cópia segura do g_best atual"""
        with self._lock:
            route = self._data["route"]
            return (
                copy.deepcopy(route) if route else None,
                self._data["cost"],
                self._data["agent_id"]
            )

    def reset(self):
        """Reseta o g_best"""
        with self._lock:
            self._data["route"] = None
            self._data["cost"] = float("inf")
            self._data["agent_id"] = None


# Singleton para uso em single-agent
# Para multi-agent, será substituído por uma instância com manager
global_best = GlobalBest()
