#!/usr/bin/env python3
"""
Unified script for running multi-agent VRP experiments.

This script consolidates all execution modes into a single parameterized interface:
- Single or multiple instance execution
- Configurable metaheuristics (mahm for all, or specific: ils, vnd, vns)
- Adaptive CSV output
- Structured logging per instance
"""

import multiprocessing as mp
import os
import csv
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any

from src.shared.blackboard import GlobalBest
from src.main import initialize_agent, run_cycle, AGENTS
from src.utils.logger import get_logger, set_instance_name
from src.agent_beliefs import AgentBeliefs
from src.utils.generate_random_feasible_route import generate_random_feasible_route
from src.utils.evaluator import evaluate_route
from src.utils.load_instance import load_instance

# Default configuration
DEFAULT_NUM_AGENTS = 5
DEFAULT_MAX_EVALUATIONS = 20000  # Default evaluation budget
DEFAULT_ACTIONS = "mahm"
NUM_RUNS = 20  # Number of repetitions for each action scenario

# Available metaheuristics
AVAILABLE_METAHEURISTICS = ["VND", "ILS", "VNS"]


def normalize_action_name(action: str) -> str:
    """
    Convert action name to uppercase standard format.
    
    Args:
        action: Action name (e.g., 'ils', 'ILS', 'Ils', 'mahm', 'MAHM')
    
    Returns:
        Normalized action name (e.g., 'ILS', 'MAHM')
    """
    action_upper = action.upper()
    if action_upper not in AVAILABLE_METAHEURISTICS and action_upper != "MAHM":
        raise ValueError(f"Invalid action '{action}'. Must be one of: 'mahm', 'ils', 'vnd', 'vns'")
    return action_upper


def get_metaheuristics_list(actions_param: str) -> List[str]:
    """
    Convert --actions parameter to list of metaheuristic names.
    
    Args:
        actions_param: Action parameter ('mahm', 'ils', 'vnd', or 'vns')
    
    Returns:
        List of metaheuristic names (e.g., ['VND', 'ILS', 'VNS'] or ['ILS'])
    """
    normalized = normalize_action_name(actions_param)
    
    if normalized == "MAHM":
        return AVAILABLE_METAHEURISTICS.copy()
    else:
        return [normalized]


def get_action_name(metaheuristics: List[str]) -> str:
    """
    Determine the action name based on the list of metaheuristics.
    
    Args:
        metaheuristics: List of metaheuristic names
    
    Returns:
        Action name: 'mahm' if all metaheuristics, otherwise the single metaheuristic name (lowercase)
    """
    if len(metaheuristics) == len(AVAILABLE_METAHEURISTICS):
        return "mahm"
    elif len(metaheuristics) == 1:
        return metaheuristics[0].lower()
    else:
        # Multiple but not all - use 'mahm' as default
        return "mahm"


def get_instance_files(instances_dir: str = "instances", instance_name: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Get instance files from the specified directory.
    
    Args:
        instances_dir: Directory containing instance files (default: 'instances')
        instance_name: Specific instance name to get (e.g., '1'). If None, returns all instances.
    
    Returns:
        List of tuples (instance_path, instance_name)
    """
    instances_path = Path(instances_dir)
    if not instances_path.exists():
        raise FileNotFoundError(f"Directory {instances_dir} not found")
    
    instance_files = []
    
    if instance_name:
        # Get specific instance
        instance_file = instances_path / f"{instance_name}.json"
        if not instance_file.exists():
            raise FileNotFoundError(f"Instance {instance_name}.json not found in {instances_dir}/")
        instance_files.append((str(instance_file), instance_name))
    else:
        # Get all instances
        for file in sorted(instances_path.glob("*.json")):
            instance_name = file.stem
            instance_files.append((str(file), instance_name))
    
    return instance_files


def initialize_agent_with_metaheuristics(agent_id: str, instance_path: str, metaheuristics: List[str]):
    """
    Initialize agent with specific metaheuristics.
    
    Args:
        agent_id: Agent identifier
        instance_path: Path to the instance file
        metaheuristics: List of metaheuristic names to use
    """
    instance = load_instance(instance_path)
    
    # Generate initial feasible route with validation
    max_attempts = 100
    initial_route = None
    feasible = False
    cost = float("inf")
    
    for attempt in range(max_attempts):
        try:
            candidate_route = generate_random_feasible_route(instance)
            feasible, candidate_cost = evaluate_route(candidate_route, instance)
            
            if feasible:
                initial_route = candidate_route
                cost = candidate_cost
                break
        except RuntimeError:
            continue
    
    if not feasible or initial_route is None:
        raise RuntimeError(
            f"[ERROR] It was not possible to generate an initial feasible route after {max_attempts} attempts"
        )
    
    beliefs = AgentBeliefs(
        agent_id=agent_id,
        metaheuristics=metaheuristics
    )
    
    beliefs.update_current_solution(initial_route, cost)
    beliefs.try_update_pbest(initial_route, cost)
    
    AGENTS[agent_id] = {
        "beliefs": beliefs,
        "instance": instance
    }


def agent_worker(agent_id: str, max_evaluations: int, num_agents: int, global_blackboard: GlobalBest, 
                 instance_path: str, instance_name: str, metaheuristics: List[str], action_name: str,
                 agent_times: Dict[str, float], agent_counters: Dict[str, int], run_number: int):
    """
    Worker function for each agent process.
    
    Args:
        agent_id: Agent ID
        max_evaluations: Maximum number of objective function evaluations (total across all agents)
        num_agents: Number of agents in the experiment
        global_blackboard: Shared GlobalBest instance
        instance_path: Path to the instance file
        instance_name: Name of the instance (for logging)
        metaheuristics: List of metaheuristics to use (restricts agent to only these)
        action_name: Name of the action (e.g., 'mahm', 'ils', 'vnd', 'vns') for logging directory
        agent_times: Shared dictionary to store execution times for each agent
        agent_counters: Shared dictionary to store evaluation counts for each agent
        run_number: Current run number (1-20)
    """
    # Inject the shared blackboard into the module
    import src.shared.blackboard
    src.shared.blackboard.global_best = global_blackboard
    
    # Set the agent_counters in compute_route_cost module for this process
    from src.utils.compute_route_cost import set_agent_counters
    set_agent_counters(agent_counters)
    
    # Set agent context for evaluation counting (must be set before any evaluations)
    from src.utils.evaluation_counter import set_agent_context
    set_agent_context(agent_id)
    
    # Set instance name, action name, and run number for logging
    set_instance_name(instance_name, action_name, run_number)
    
    # Initialize logger for this agent
    logger = get_logger(agent_id, instance_name, action_name, run_number)
    
    # Start timing
    start_time = time.time()
    
    logger.log("===================================")
    logger.log(" STARTING AGENT EXECUTION ")
    logger.log("===================================")
    logger.log(f"Instance: {instance_name}")
    
    # Initialize the agent with the specified instance and metaheuristics
    initialize_agent_with_metaheuristics(agent_id, instance_path, metaheuristics)
    
    # Verify that the agent was initialized with the correct metaheuristics
    agent_beliefs = AGENTS[agent_id]["beliefs"]
    available_actions = list(agent_beliefs.actions.keys())
    if set(available_actions) != set(metaheuristics):
        raise ValueError(
            f"[ERROR] Agent {agent_id} was not initialized correctly. "
            f"Expected: {metaheuristics}, Got: {available_actions}"
        )
    
    logger.log(f"Available Metaheuristics: {available_actions}")
    
    beliefs = AGENTS[agent_id]["beliefs"]
    
    # Calculate per-agent evaluation budget
    agent_budget = max_evaluations // num_agents
    beliefs.set_evaluation_budget(agent_budget)
    
    logger.log("\n--- Initial state ---")
    logger.log(f"Initial Solution/Route: {beliefs.current_route}")
    logger.log(f"Initial Cost: {beliefs.current_cost}")
    logger.log(f"Evaluation Budget: {agent_budget}")
    logger.log("----------------------\n")
    
    # Sync evaluation count from shared counter (accounts for initialization evaluations)
    initial_count = agent_counters.get(agent_id, 0)
    beliefs.update_evaluation_count(initial_count)
    
    # Execute the agent cycle with evaluation budget stopping criterion
    iteration = 0
    while beliefs.has_budget_remaining():
        logger.log(f"\n==== ITERATION {iteration} ====")
        
        run_cycle(agent_id)
        
        # Update evaluation count from shared counter
        current_count = agent_counters.get(agent_id, 0)
        beliefs.update_evaluation_count(current_count)
        
        g_route, g_cost, g_agent = global_blackboard.get()
        logger.log_state(
            current_cost=beliefs.current_cost,
            p_best_cost=beliefs.p_best_cost,
            g_best_cost=g_cost if g_route is not None else None,
            g_best_agent=g_agent if g_route is not None else None
        )
        
        iteration += 1
        
        # Check budget again after updating count
        if not beliefs.has_budget_remaining():
            break
    
    logger.log("\n===================================")
    logger.log(" EXECUTION FINALIZED ")
    logger.log("===================================")
    
    logger.log("\n--- Final result ---")
    logger.log(f"Agent's p_best cost: {beliefs.p_best_cost}")
    logger.log(f"Agent's p_best solution/route: {beliefs.p_best_route}")
    
    g_route, g_cost, g_agent = global_blackboard.get()
    logger.log(f"\nGlobal Best:")
    logger.log(f"Cost : {g_cost}")
    logger.log(f"Solution/Route  : {g_route}")
    
    # Calculate and log execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.log(f"\nRun Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Get final evaluation count and log it
    final_evaluation_count = agent_counters.get(agent_id, 0)
    logger.log(f"Total Evaluations: {final_evaluation_count}")
    
    # Store execution time in shared dictionary
    agent_times[agent_id] = elapsed_time


def write_outcome_log(instance_name: str, action_name: str, num_agents: int, 
                      g_best_cost: Optional[float], g_best_agent: Optional[str], 
                      total_time: float, agent_counters: Dict[str, int], run_number: int):
    """
    Write outcome log file with experiment summary.
    
    Args:
        instance_name: Name of the instance
        action_name: Name of the action (e.g., 'mahm', 'ils', 'vnd', 'vns')
        num_agents: Number of agents used
        g_best_cost: Best cost found (global best)
        g_best_agent: Agent ID that found the global best
        total_time: Total execution time (sum of all agent times)
        agent_counters: Shared dictionary with evaluation counts for each agent
        run_number: Current run number (1-20)
    """
    log_dir = f"logs/{instance_name}/{action_name.lower()}/{run_number}"
    outcome_file = f"{log_dir}/outcome.log"
    
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    with open(outcome_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" EXPERIMENT OUTCOME SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Instance: {instance_name}\n")
        f.write(f"Action: {action_name.upper()}\n")
        f.write(f"Run Number: {run_number}\n")
        f.write(f"Number of Agents: {num_agents}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("GLOBAL BEST SOLUTION\n")
        f.write("-" * 80 + "\n")
        if g_best_cost is not None:
            f.write(f"Best Cost: {g_best_cost:.2f}\n")
            if g_best_agent is not None:
                f.write(f"Found by Agent: {g_best_agent}\n")
            else:
                f.write("Found by Agent: N/A\n")
        else:
            f.write("Best Cost: N/A (No solution found)\n")
            f.write("Found by Agent: N/A\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("EXECUTION TIME\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"Average Time per Agent: {total_time/num_agents:.2f} seconds\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("OBJECTIVE FUNCTION EVALUATIONS\n")
        f.write("-" * 80 + "\n")
        # Calculate total evaluations across all agents
        total_evaluations = sum(agent_counters.values()) if agent_counters else 0
        f.write(f"Total Evaluations: {total_evaluations}\n")
        f.write(f"Average Evaluations per Agent: {total_evaluations/num_agents:.2f}\n")
        f.write("\nPer-Agent Breakdown:\n")
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            agent_eval_count = agent_counters.get(agent_id, 0)
            f.write(f"  {agent_id}: {agent_eval_count} evaluations\n")
        
        f.write("\n" + "=" * 80 + "\n")


def parse_outcome_log(outcome_file: str) -> Dict[str, Any]:
    """
    Parse an outcome.log file to extract key information.
    
    Args:
        outcome_file: Path to outcome.log file
    
    Returns:
        Dictionary with parsed information (g_best_cost, total_time, total_evaluations)
    """
    result = {
        "g_best_cost": None,
        "total_time": None,
        "total_evaluations": None
    }
    
    if not os.path.exists(outcome_file):
        return result
    
    with open(outcome_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if "Best Cost:" in line:
            try:
                # Extract cost value
                parts = line.split("Best Cost:")
                if len(parts) > 1:
                    cost_str = parts[1].strip()
                    if cost_str != "N/A (No solution found)":
                        result["g_best_cost"] = float(cost_str)
            except (ValueError, IndexError):
                pass
        elif "Total Time:" in line:
            try:
                # Extract time value (format: "Total Time: 225.45 seconds")
                parts = line.split("Total Time:")
                if len(parts) > 1:
                    time_str = parts[1].strip().split()[0]  # Get number before "seconds"
                    result["total_time"] = float(time_str)
            except (ValueError, IndexError):
                pass
        elif "Total Evaluations:" in line:
            try:
                # Extract evaluation count
                parts = line.split("Total Evaluations:")
                if len(parts) > 1:
                    result["total_evaluations"] = int(parts[1].strip())
            except (ValueError, IndexError):
                pass
    
    return result


def write_summary_log(instance_name: str, action_name: str, num_agents: int, num_runs: int):
    """
    Write summary log file consolidating results from all runs.
    
    Args:
        instance_name: Name of the instance
        action_name: Name of the action (e.g., 'mahm', 'ils', 'vnd', 'vns')
        num_agents: Number of agents used
        num_runs: Number of runs (should be 20)
    """
    log_dir = f"logs/{instance_name}/{action_name.lower()}"
    summary_file = f"{log_dir}/summary.log"
    
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Collect data from all runs
    g_best_costs = []
    total_times = []
    total_evaluations_list = []
    
    for run_num in range(1, num_runs + 1):
        outcome_file = f"{log_dir}/{run_num}/outcome.log"
        parsed = parse_outcome_log(outcome_file)
        
        if parsed["g_best_cost"] is not None:
            g_best_costs.append(parsed["g_best_cost"])
        if parsed["total_time"] is not None:
            total_times.append(parsed["total_time"])
        if parsed["total_evaluations"] is not None:
            total_evaluations_list.append(parsed["total_evaluations"])
    
    # Calculate statistics
    avg_g_best_cost = sum(g_best_costs) / len(g_best_costs) if g_best_costs else None
    total_execution_time = sum(total_times) if total_times else 0.0
    avg_time_per_run = total_execution_time / len(total_times) if total_times else 0.0
    total_evaluations_all_runs = sum(total_evaluations_list) if total_evaluations_list else 0
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" EXPERIMENT SUMMARY (20 RUNS)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Instance: {instance_name}\n")
        f.write(f"Action: {action_name.upper()}\n")
        f.write(f"Number of Agents: {num_agents}\n")
        f.write(f"Number of Runs: {num_runs}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("GLOBAL BEST SOLUTION (AVERAGE)\n")
        f.write("-" * 80 + "\n")
        if avg_g_best_cost is not None:
            f.write(f"Average Best Cost: {avg_g_best_cost:.2f}\n")
            f.write(f"Runs with valid solutions: {len(g_best_costs)}/{num_runs}\n")
        else:
            f.write("Average Best Cost: N/A (No valid solutions found)\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("EXECUTION TIME\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Time (all runs): {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)\n")
        f.write(f"Average Time per Run: {avg_time_per_run:.2f} seconds ({avg_time_per_run/60:.2f} minutes)\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("OBJECTIVE FUNCTION EVALUATIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Evaluations (all runs): {total_evaluations_all_runs}\n")
        if total_evaluations_list:
            avg_evaluations_per_run = total_evaluations_all_runs / len(total_evaluations_list)
            f.write(f"Average Evaluations per Run: {avg_evaluations_per_run:.2f}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("PER-RUN BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        for run_num in range(1, num_runs + 1):
            outcome_file = f"{log_dir}/{run_num}/outcome.log"
            parsed = parse_outcome_log(outcome_file)
            g_cost = parsed["g_best_cost"] if parsed["g_best_cost"] is not None else "N/A"
            run_time = parsed["total_time"] if parsed["total_time"] is not None else "N/A"
            f.write(f"Run {run_num:2d}: g_best={g_cost:>10} | time={run_time:>10} seconds\n")
        
        f.write("\n" + "=" * 80 + "\n")


def run_experiment_for_instance(instance_path: str, instance_name: str, metaheuristics: List[str], 
                                num_agents: int, max_evaluations: int, run_number: int) -> Optional[float]:
    """
    Run experiment for a specific instance and metaheuristic configuration.
    
    Args:
        instance_path: Path to the instance file
        instance_name: Name of the instance
        metaheuristics: List of metaheuristics to use
        num_agents: Number of agents to run
        max_evaluations: Maximum number of objective function evaluations (total across all agents)
        run_number: Current run number (1-20)
    
    Returns:
        g_best_cost: Best cost found, or None if no solution found
    """
    mp.set_start_method("spawn", force=True)
    
    manager = mp.Manager()
    global_blackboard = GlobalBest(manager)
    agent_times = manager.dict()  # Shared dictionary for agent execution times
    agent_counters = manager.dict()  # Shared dictionary for agent evaluation counts
    
    # Determine action name for logging directory
    action_name = get_action_name(metaheuristics)
    
    processes = []
    
    # Create and start processes for each agent
    for i in range(num_agents):
        p = mp.Process(
            target=agent_worker,
            args=(f"agent_{i}", max_evaluations, num_agents, global_blackboard, instance_path, instance_name, metaheuristics, action_name, agent_times, agent_counters, run_number)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Get final result
    g_route, g_cost, g_agent = global_blackboard.get()
    
    # Calculate total execution time (sum of all agent times)
    total_time = sum(agent_times.values()) if agent_times else 0.0
    
    # Write outcome log file
    write_outcome_log(instance_name, action_name, num_agents, g_cost, g_agent, total_time, agent_counters, run_number)
    
    # Clear AGENTS registry for next experiment
    AGENTS.clear()
    
    return g_cost if g_route is not None else None


def read_existing_csv(csv_filename: str) -> Dict[str, Dict[str, str]]:
    """
    Read existing CSV file if it exists.
    
    Args:
        csv_filename: Path to CSV file
    
    Returns:
        Dictionary mapping instance names to their results (column -> value)
    """
    existing_data = {}
    if Path(csv_filename).exists():
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                instance_name = row['Instance']
                existing_data[instance_name] = {k: v for k, v in row.items() if k != 'Instance'}
    return existing_data


def write_adaptive_csv(results: Dict[str, Dict[str, Union[Optional[float], int, str]]], csv_filename: str, 
                       executed_metaheuristics: List[str]):
    """
    Write CSV with adaptive columns based on executed metaheuristics.
    
    Args:
        results: Dictionary mapping instance names to their results (metaheuristic -> cost, "# vertices" -> num_nodes)
        csv_filename: Path to CSV file
        executed_metaheuristics: List of metaheuristics that were executed
    """
    # Read existing CSV to preserve previous data
    existing_data = read_existing_csv(csv_filename)
    
    # Determine column name based on executed metaheuristics
    if len(executed_metaheuristics) == len(AVAILABLE_METAHEURISTICS):
        # All metaheuristics were executed, use "MAHM" column
        column_name = "MAHM"
    elif len(executed_metaheuristics) == 1:
        # Single metaheuristic, use its name
        column_name = executed_metaheuristics[0]
    else:
        # Multiple but not all - use comma-separated names or "MAHM"
        # For simplicity, use "MAHM" if multiple are executed
        column_name = "MAHM"
    
    # Update existing data with new results
    for instance_name, instance_results in results.items():
        if instance_name not in existing_data:
            existing_data[instance_name] = {}
        
        # Update the column for this execution
        cost = instance_results.get(column_name)
        if cost is not None:
            existing_data[instance_name][column_name] = f"{cost:.2f}"
        else:
            existing_data[instance_name][column_name] = "N/A"
        
        # Update "# vertices" column
        num_vertices = instance_results.get("# vertices")
        if num_vertices is not None:
            existing_data[instance_name]["# vertices"] = str(num_vertices)
    
    # Determine all columns (Instance + # vertices + all unique metaheuristic columns)
    all_columns = set(['Instance', '# vertices'])
    for instance_data in existing_data.values():
        all_columns.update(instance_data.keys())
    
    # Sort columns: Instance first, then "# vertices", then alphabetically
    sorted_columns = ['Instance', '# vertices'] + sorted([c for c in all_columns if c not in ['Instance', '# vertices']])
    
    # Write CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted_columns)
        writer.writeheader()
        
        # Sort instances numerically
        sorted_instances = sorted(existing_data.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        for instance_name in sorted_instances:
            row = {'Instance': instance_name}
            row.update(existing_data[instance_name])
            # Fill missing columns with "N/A"
            for col in sorted_columns:
                if col != 'Instance' and col not in row:
                    row[col] = "N/A"
            writer.writerow(row)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified multi-agent VRP experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all instances with all metaheuristics (MAHM - default)
  python run.py
  
  # Run specific instance with all metaheuristics (MAHM)
  python run.py --instance 1
  
  # Run all instances with only ILS
  python run.py --actions ils
  
  # Run specific instance with VND, 4 agents, 10000 evaluations
  python run.py --instance 50 --actions vnd --n-agents 4 --max-evaluations 10000
        """
    )
    
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Instance name from instances/ directory (e.g., '1'). If not provided, runs all instances."
    )
    
    parser.add_argument(
        "--n-agents",
        type=int,
        default=DEFAULT_NUM_AGENTS,
        help=f"Number of parallel agents (default: {DEFAULT_NUM_AGENTS})"
    )
    
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=DEFAULT_MAX_EVALUATIONS,
        help=f"Maximum number of objective function evaluations (total across all agents) (default: {DEFAULT_MAX_EVALUATIONS})"
    )
    
    parser.add_argument(
        "--actions",
        type=str,
        default=None,
        help="Metaheuristic selection: 'mahm', 'ils', 'vnd', or 'vns'. If not provided, runs all 4 actions with 20 repetitions each."
    )
    
    args = parser.parse_args()
    
    # Get instance files
    try:
        instance_files = get_instance_files("instances", args.instance)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if not instance_files:
        print("No instance files found in 'instances/'!")
        return
    
    # Determine which actions to run
    if args.actions:
        # Single action mode (backward compatibility)
        try:
            metaheuristics = get_metaheuristics_list(args.actions)
            actions_to_run = [args.actions]
        except ValueError as e:
            print(f"Error: {e}")
            return
    else:
        # Run all 4 actions with 20 repetitions each
        actions_to_run = ["mahm", "ils", "vnd", "vns"]
    
    print("=" * 80)
    print(" UNIFIED MULTI-AGENT VRP EXPERIMENT")
    print("=" * 80)
    print(f"Number of agents: {args.n_agents}")
    print(f"Max evaluations (total): {args.max_evaluations}")
    print(f"Actions to run: {actions_to_run}")
    print(f"Runs per action: {NUM_RUNS}")
    print(f"Instances to process: {len(instance_files)}")
    if args.instance:
        print(f"Specific instance: {args.instance}")
    print("=" * 80)
    print()
    
    # Results dictionary: {instance_name: {action_name: average_cost, "# vertices": num_nodes}}
    results = {}
    
    # Run experiments for each action
    for action_name in actions_to_run:
        print("\n" + "=" * 80)
        print(f" RUNNING ACTION: {action_name.upper()}")
        print("=" * 80)
        
        # Get metaheuristics for this action
        try:
            metaheuristics = get_metaheuristics_list(action_name)
        except ValueError as e:
            print(f"Error: {e}")
            continue
        
        # Determine column name for CSV
        if len(metaheuristics) == len(AVAILABLE_METAHEURISTICS):
            column_name = "MAHM"
        elif len(metaheuristics) == 1:
            column_name = metaheuristics[0]
        else:
            column_name = "MAHM"
        
        # Run experiments for each instance
        for idx, (instance_path, instance_name) in enumerate(instance_files, 1):
            print(f"\n[{idx}/{len(instance_files)}] Instance {instance_name} - Action {action_name.upper()}")
            
            # Initialize results for this instance if not exists
            if instance_name not in results:
                try:
                    instance = load_instance(instance_path)
                    num_nodes = instance.get("num_nodes", "N/A")
                    results[instance_name] = {
                        "# vertices": num_nodes
                    }
                except:
                    results[instance_name] = {
                        "# vertices": "N/A"
                    }
            
            # Run 20 times for this instance/action combination
            g_best_costs = []
            
            for run_num in range(1, NUM_RUNS + 1):
                print(f"  Run {run_num}/{NUM_RUNS}...", end=" ", flush=True)
                
                try:
                    g_cost = run_experiment_for_instance(
                        instance_path, instance_name, metaheuristics, args.n_agents, args.max_evaluations, run_num
                    )
                    
                    if g_cost is not None:
                        g_best_costs.append(g_cost)
                        print(f"✓ Cost: {g_cost:.2f}")
                    else:
                        print(f"❌ No solution")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate average cost for this instance/action
            if g_best_costs:
                avg_cost = sum(g_best_costs) / len(g_best_costs)
                results[instance_name][column_name] = avg_cost
                print(f"  Average cost: {avg_cost:.2f} (from {len(g_best_costs)}/{NUM_RUNS} successful runs)")
            else:
                results[instance_name][column_name] = None
                print(f"  Average cost: N/A (no successful runs)")
            
            # Write summary log after all 20 runs for this instance/action
            write_summary_log(instance_name, action_name, args.n_agents, NUM_RUNS)
    
    # Write adaptive CSV
    print("\n" + "=" * 80)
    print(" GENERATING RESULTS TABLE")
    print("=" * 80)
    
    csv_filename = "results.csv"
    # Collect all executed metaheuristics (all available for CSV column determination)
    write_adaptive_csv(results, csv_filename, AVAILABLE_METAHEURISTICS)
    
    print(f"\n✓ Tabela salva em: {csv_filename}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print(" RESUMO DOS RESULTADOS")
    print("=" * 80)
    
    # Read final CSV to show all columns
    existing_data = read_existing_csv(csv_filename)
    if existing_data:
        # Get all columns
        all_columns = set(['Instance', '# vertices'])
        for instance_data in existing_data.values():
            all_columns.update(instance_data.keys())
        # Sort columns: Instance first, then "# vertices", then alphabetically
        sorted_columns = ['Instance', '# vertices'] + sorted([c for c in all_columns if c not in ['Instance', '# vertices']])
        
        # Print header
        for col in sorted_columns:
            if col == 'Instance':
                print(f"{col:<12}", end="")
            elif col == '# vertices':
                print(f"{col:>12}", end="")
            else:
                print(f"{col:>15}", end="")
        print()
        print("-" * 80)
        
        # Print rows
        sorted_instances = sorted(existing_data.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        for instance_name in sorted_instances:
            instance_data = existing_data[instance_name]
            for col in sorted_columns:
                if col == 'Instance':
                    print(f"{instance_name:<12}", end="")
                elif col == '# vertices':
                    value = instance_data.get(col, "N/A")
                    print(f"{value:>12}", end="")
                else:
                    value = instance_data.get(col, "N/A")
                    if value != "N/A":
                        try:
                            print(f"{float(value):>15.2f}", end="")
                        except (ValueError, TypeError):
                            print(f"{value:>15}", end="")
                    else:
                        print(f"{'N/A':>15}", end="")
            print()
    
    print("\n" + "=" * 80)
    print(" EXPERIMENTO CONCLUÍDO")
    print("=" * 80)
    print(f"\nLogs organizados em: logs/{{instance_name}}/agent_X.log")


if __name__ == "__main__":
    main()

