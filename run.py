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
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_ACTIONS = "mahm"

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


def agent_worker(agent_id: str, max_iterations: int, global_blackboard: GlobalBest, 
                 instance_path: str, instance_name: str, metaheuristics: List[str], action_name: str):
    """
    Worker function for each agent process.
    
    Args:
        agent_id: Agent ID
        max_iterations: Maximum number of iterations
        global_blackboard: Shared GlobalBest instance
        instance_path: Path to the instance file
        instance_name: Name of the instance (for logging)
        metaheuristics: List of metaheuristics to use (restricts agent to only these)
        action_name: Name of the action (e.g., 'mahm', 'ils', 'vnd', 'vns') for logging directory
    """
    # Inject the shared blackboard into the module
    import src.shared.blackboard
    src.shared.blackboard.global_best = global_blackboard
    
    # Set instance name and action name for logging
    set_instance_name(instance_name, action_name)
    
    # Initialize logger for this agent
    logger = get_logger(agent_id, instance_name, action_name)
    
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
    
    logger.log("\n--- Initial state ---")
    logger.log(f"Initial Solution/Route: {beliefs.current_route}")
    logger.log(f"Initial Cost: {beliefs.current_cost}")
    logger.log("----------------------\n")
    
    # Execute the agent cycle
    for iteration in range(max_iterations):
        logger.log(f"\n==== ITERATION {iteration} ====")
        
        run_cycle(agent_id)
        
        g_route, g_cost, g_agent = global_blackboard.get()
        logger.log_state(
            current_cost=beliefs.current_cost,
            p_best_cost=beliefs.p_best_cost,
            g_best_cost=g_cost if g_route is not None else None,
            g_best_agent=g_agent if g_route is not None else None
        )
    
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


def run_experiment_for_instance(instance_path: str, instance_name: str, metaheuristics: List[str], 
                                num_agents: int, max_iterations: int) -> Optional[float]:
    """
    Run experiment for a specific instance and metaheuristic configuration.
    
    Args:
        instance_path: Path to the instance file
        instance_name: Name of the instance
        metaheuristics: List of metaheuristics to use
        num_agents: Number of agents to run
        max_iterations: Maximum number of iterations per agent
    
    Returns:
        g_best_cost: Best cost found, or None if no solution found
    """
    mp.set_start_method("spawn", force=True)
    
    manager = mp.Manager()
    global_blackboard = GlobalBest(manager)
    
    # Determine action name for logging directory
    action_name = get_action_name(metaheuristics)
    
    processes = []
    
    # Create and start processes for each agent
    for i in range(num_agents):
        p = mp.Process(
            target=agent_worker,
            args=(f"agent_{i}", max_iterations, global_blackboard, instance_path, instance_name, metaheuristics, action_name)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Get final result
    g_route, g_cost, g_agent = global_blackboard.get()
    
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
  
  # Run specific instance with VND, 4 agents, 50 iterations
  python run.py --instance 50 --actions vnd --n-agents 4 --max-iterations 50
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
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum iterations per agent (default: {DEFAULT_MAX_ITERATIONS})"
    )
    
    parser.add_argument(
        "--actions",
        type=str,
        default=DEFAULT_ACTIONS,
        help="Metaheuristic selection: 'mahm', 'ils', 'vnd', or 'vns' (default: 'mahm')"
    )
    
    args = parser.parse_args()
    
    # Validate and normalize actions parameter
    try:
        metaheuristics = get_metaheuristics_list(args.actions)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Get instance files
    try:
        instance_files = get_instance_files("instances", args.instance)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if not instance_files:
        print("No instance files found in 'instances/'!")
        return
    
    # Determine column name for CSV
    if len(metaheuristics) == len(AVAILABLE_METAHEURISTICS):
        column_name = "MAHM"
    elif len(metaheuristics) == 1:
        column_name = metaheuristics[0]
    else:
        column_name = "MAHM"
    
    print("=" * 80)
    print(" UNIFIED MULTI-AGENT VRP EXPERIMENT")
    print("=" * 80)
    print(f"Number of agents: {args.n_agents}")
    print(f"Max iterations per agent: {args.max_iterations}")
    print(f"Actions: {args.actions} -> {metaheuristics}")
    print(f"Instances to process: {len(instance_files)}")
    if args.instance:
        print(f"Specific instance: {args.instance}")
    print("=" * 80)
    print()
    
    # Results dictionary: {instance_name: {column_name: cost, "# vertices": num_nodes}}
    results = {}
    
    # Run experiments
    for idx, (instance_path, instance_name) in enumerate(instance_files, 1):
        print(f"[{idx}/{len(instance_files)}] Executando instância {instance_name}...", end=" ", flush=True)
        
        try:
            # Load instance to get num_nodes
            instance = load_instance(instance_path)
            num_nodes = instance.get("num_nodes", "N/A")
            
            g_cost = run_experiment_for_instance(
                instance_path, instance_name, metaheuristics, args.n_agents, args.max_iterations
            )
            
            results[instance_name] = {
                column_name: g_cost,
                "# vertices": num_nodes
            }
            
            if g_cost is not None:
                print(f"✓ Custo: {g_cost}")
            else:
                print(f"❌ Nenhuma solução encontrada")
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            # Try to get num_nodes even if experiment failed
            try:
                instance = load_instance(instance_path)
                num_nodes = instance.get("num_nodes", "N/A")
            except:
                num_nodes = "N/A"
            results[instance_name] = {
                column_name: None,
                "# vertices": num_nodes
            }
            import traceback
            traceback.print_exc()
    
    # Write adaptive CSV
    print("\n" + "=" * 80)
    print(" GERANDO TABELA DE RESULTADOS")
    print("=" * 80)
    
    csv_filename = "results.csv"
    write_adaptive_csv(results, csv_filename, metaheuristics)
    
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

