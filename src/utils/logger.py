import os
from typing import Optional

class AgentLogger:
    """
    Logger simples para cada agente que escreve em arquivo.
    Cada agente tem seu próprio arquivo de log: logs/{instance_name}/{agent_id}.log
    """
    
    def __init__(self, agent_id: str, instance_name: Optional[str] = None):
        self.agent_id = agent_id
        if instance_name:
            self.log_dir = f"logs/{instance_name}"
        else:
            self.log_dir = "logs"
        self.log_file = f"{self.log_dir}/{agent_id}.log"
        
        # Criar diretório de logs se não existir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Limpar arquivo de log ao inicializar (opcional - pode ser append)
        # Comentado para manter histórico
        # if os.path.exists(self.log_file):
        #     open(self.log_file, 'w').close()
    
    def log(self, message: str):
        """Escreve uma mensagem no arquivo de log do agente"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
    
    def log_iteration(self, iteration: int, message: str):
        """Escreve uma mensagem com número de iteração"""
        self.log(f"Iteration {iteration}: {message}")
    
    def log_phase(self, phase: str, message: str = ""):
        """Escreve uma mensagem de fase do ciclo"""
        if message:
            self.log(f"{phase}: {message}")
        else:
            self.log(phase)
    
    def log_state(self, current_cost: float, p_best_cost: float, g_best_cost: Optional[float] = None, g_best_agent: Optional[str] = None):
        """Escreve estado atual do agente"""
        state_msg = f"Current Cost: {current_cost}, p_best: {p_best_cost}"
        if g_best_cost is not None:
            state_msg += f", g_best: {g_best_cost} (agent {g_best_agent})"
        self.log(state_msg)


# Singleton global - será inicializado por cada agente
_loggers: dict[str, AgentLogger] = {}
_current_instance: Optional[str] = None

def set_instance_name(instance_name: str):
    """Define o nome da instância atual para os logs"""
    global _current_instance, _loggers
    _current_instance = instance_name
    # Limpar loggers existentes quando mudar de instância
    _loggers.clear()

def get_logger(agent_id: str, instance_name: Optional[str] = None) -> AgentLogger:
    """
    Obtém ou cria um logger para o agente.
    
    Args:
        agent_id: ID do agente
        instance_name: Nome da instância (se None, usa _current_instance)
    """
    global _current_instance
    # Usar instance_name fornecido ou o global
    instance = instance_name if instance_name is not None else _current_instance
    
    # Criar chave única combinando instance e agent_id
    logger_key = f"{instance}_{agent_id}" if instance else agent_id
    
    if logger_key not in _loggers:
        _loggers[logger_key] = AgentLogger(agent_id, instance)
    return _loggers[logger_key]

