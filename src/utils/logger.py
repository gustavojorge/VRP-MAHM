import os
from typing import Optional

class AgentLogger:
    """
    Simple logger for each agent that writes to file.
    Each agent has its own log file: logs/{instance_name}/{action_name}/{run_number}/{agent_id}.log
    """
    
    def __init__(self, agent_id: str, instance_name: Optional[str] = None, action_name: Optional[str] = None, run_number: Optional[int] = None):
        self.agent_id = agent_id
        if instance_name:
            if action_name:
                if run_number is not None:
                    self.log_dir = f"logs/{instance_name}/{action_name.lower()}/{run_number}"
                else:
                    self.log_dir = f"logs/{instance_name}/{action_name.lower()}"
            else:
                self.log_dir = f"logs/{instance_name}"
        else:
            self.log_dir = "logs"
        self.log_file = f"{self.log_dir}/{agent_id}.log"
        
        # Create log directory if it does not exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Clear log file when initializing (optional - can be append)
        # Commented to keep history
        # if os.path.exists(self.log_file):
        #     open(self.log_file, 'w').close()
    
    def log(self, message: str):
        """Writes a message to the agent's log file"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
    
    def log_iteration(self, iteration: int, message: str):
        """Writes a message with iteration number"""
        self.log(f"Iteration {iteration}: {message}")
    
    def log_phase(self, phase: str, message: str = ""):
        """Writes a message of the cycle phase"""
        if message:
            self.log(f"{phase}: {message}")
        else:
            self.log(phase)
    
    def log_state(self, current_cost: float, p_best_cost: float, g_best_cost: Optional[float] = None, g_best_agent: Optional[str] = None):
        """Writes the current state of the agent"""
        state_msg = f"Current Cost: {current_cost}, p_best: {p_best_cost}"
        if g_best_cost is not None:
            state_msg += f", g_best: {g_best_cost} (agent {g_best_agent})"
        self.log(state_msg)


# Singleton global - will be initialized by each agent
_loggers: dict[str, AgentLogger] = {}
_current_instance: Optional[str] = None
_current_action: Optional[str] = None
_current_run: Optional[int] = None

def set_instance_name(instance_name: str, action_name: Optional[str] = None, run_number: Optional[int] = None):
    """Define the current instance name, action name, and run number for the logs"""
    global _current_instance, _current_action, _current_run, _loggers
    _current_instance = instance_name
    _current_action = action_name
    _current_run = run_number
    # Clear existing loggers when changing instance, action, or run
    _loggers.clear()

def get_logger(agent_id: str, instance_name: Optional[str] = None, action_name: Optional[str] = None, run_number: Optional[int] = None) -> AgentLogger:
    """
    Gets or creates a logger for the agent.
    
    Args:
        agent_id: Agent ID
        instance_name: Instance name (if None, uses _current_instance)
        action_name: Action name (if None, uses _current_action)
        run_number: Run number (if None, uses _current_run)
    """
    global _current_instance, _current_action, _current_run
    # Use the provided values or the global
    instance = instance_name if instance_name is not None else _current_instance
    action = action_name if action_name is not None else _current_action
    run = run_number if run_number is not None else _current_run
    
    # Create a unique key combining instance, action, run, and agent_id
    if instance and action and run is not None:
        logger_key = f"{instance}_{action}_{run}_{agent_id}"
    elif instance and action:
        logger_key = f"{instance}_{action}_{agent_id}"
    elif instance:
        logger_key = f"{instance}_{agent_id}"
    else:
        logger_key = agent_id
    
    if logger_key not in _loggers:
        _loggers[logger_key] = AgentLogger(agent_id, instance, action, run)
    return _loggers[logger_key]

