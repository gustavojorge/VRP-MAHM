"""
Checkpoint module for recording experiment evolution.

This module provides functionality to record the evolution of p_best and g_best
at regular intervals (every 100,000 global evaluations) in a CSV file.
"""

import os
import csv
from typing import Dict, Optional, Any

# Module-level variables for shared structures
_agent_p_bests: Optional[Dict[str, Dict[str, Any]]] = None
_checkpoint_lock: Optional[Any] = None  # Can be threading.Lock or multiprocessing.Lock


def set_shared_structures(agent_p_bests: Dict[str, Dict[str, Any]], checkpoint_lock: Any) -> None:
    """
    Set the shared structures for checkpoint management.
    
    Args:
        agent_p_bests: Shared dictionary mapping agent_id to {"cost": float, "route": List[int]}
        checkpoint_lock: Lock for thread-safe checkpoint writing (threading.Lock or multiprocessing.Lock)
    """
    global _agent_p_bests, _checkpoint_lock
    _agent_p_bests = agent_p_bests
    _checkpoint_lock = checkpoint_lock


def update_agent_p_best(agent_id: str, cost: float, route: list) -> None:
    """
    Update the p_best for a specific agent in the shared dictionary.
    
    Args:
        agent_id: Agent identifier
        cost: Best cost found by the agent
        route: Best route found by the agent
    """
    global _agent_p_bests
    if _agent_p_bests is not None:
        _agent_p_bests[agent_id] = {
            "cost": cost,
            "route": route.copy() if route else None
        }


def write_checkpoint(
    instance_name: str,
    action_name: str,
    run_number: int,
    total_evaluations: int,
    global_blackboard: Any,
    num_agents: int
) -> None:
    """
    Write a checkpoint to CSV if total_evaluations is a multiple of 100,000.
    
    This function checks if a checkpoint should be written (every 100,000 evaluations)
    and writes it to the checkpoint.csv file. It uses a lock to ensure only one
    process writes the checkpoint.
    
    Args:
        instance_name: Name of the instance
        action_name: Name of the action (e.g., 'mahm', 'ils', 'vnd', 'vns')
        run_number: Current run number
        total_evaluations: Total number of global evaluations
        global_blackboard: GlobalBest instance to get g_best
        num_agents: Number of agents in the experiment
    """
    global _checkpoint_lock, _agent_p_bests
    
    if _checkpoint_lock is None:
        return  # Checkpoint not initialized
    
    # Check if we should write a checkpoint (multiple of 100,000)
    checkpoint_interval = 20000
    
    # Only proceed if total_evaluations is at least 100,000
    if total_evaluations < checkpoint_interval:
        return
    
    # Prepare CSV file path
    log_dir = f"logs/{instance_name}/{action_name.lower()}/{run_number}"
    checkpoint_file = f"{log_dir}/checkpoint.csv"
    
    # Acquire lock to ensure only one process writes
    with _checkpoint_lock:
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(checkpoint_file)
        
        # Find the last written checkpoint
        last_written_checkpoint = 0
        if file_exists:
            try:
                with open(checkpoint_file, "r", newline="", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        checkpoint_val = int(row.get("checkpoint_evaluations", 0))
                        last_written_checkpoint = max(last_written_checkpoint, checkpoint_val)
            except (ValueError, KeyError, IOError):
                # If file is corrupted or can't be read, start from 0
                last_written_checkpoint = 0
        
        # Calculate the next checkpoint that should be written
        # It should be the next multiple of 100,000 after the last written checkpoint
        next_checkpoint = ((last_written_checkpoint // checkpoint_interval) + 1) * checkpoint_interval
        
        # Only write if we've reached or exceeded the next checkpoint
        if total_evaluations < next_checkpoint:
            return
        
        # Check if this checkpoint was already written (safety check)
        # This can happen if multiple processes check simultaneously
        if file_exists:
            try:
                with open(checkpoint_file, "r", newline="", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        checkpoint_val = int(row.get("checkpoint_evaluations", 0))
                        if checkpoint_val == next_checkpoint:
                            # This checkpoint was already written by another process
                            return
            except (ValueError, KeyError, IOError):
                # If file is corrupted or can't be read, proceed to write
                pass
        
        # Use next_checkpoint as the current_checkpoint to write
        current_checkpoint = next_checkpoint
        
        # Get g_best from blackboard
        g_route, g_cost, g_agent = global_blackboard.get()
        
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Prepare row data
        row_data = {
            "run_id": run_number,
            "checkpoint_evaluations": current_checkpoint,
            "g_best_cost": f"{g_cost:.2f}" if g_route is not None else "inf",
            "g_best_agent": g_agent if g_route is not None else "N/A"
        }
        
        # Add p_best for each agent
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            if _agent_p_bests is not None and agent_id in _agent_p_bests:
                p_best_data = _agent_p_bests[agent_id]
                p_best_cost = p_best_data.get("cost", float("inf"))
                row_data[f"p_best_agent_{i}"] = f"{p_best_cost:.2f}" if p_best_cost != float("inf") else "inf"
            else:
                row_data[f"p_best_agent_{i}"] = "inf"
        
        # Write to CSV
        fieldnames = ["run_id", "checkpoint_evaluations", "g_best_cost", "g_best_agent"]
        fieldnames.extend([f"p_best_agent_{i}" for i in range(num_agents)])
        
        with open(checkpoint_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write row
            writer.writerow(row_data)

