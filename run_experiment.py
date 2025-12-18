#!/usr/bin/env python3
"""
Script para executar experimentos comparativos entre diferentes configurações de metaheurísticas.

Variações:
1. Todas as metaheurísticas (VND, ILS, VNS)
2. Apenas ILS
3. Apenas VND
4. Apenas VNS
"""

import multiprocessing as mp
import os
import csv
from pathlib import Path
from src.shared.blackboard import GlobalBest
from src.main import initialize_agent, run_cycle, AGENTS
from src.utils.logger import get_logger, set_instance_name

# Configuração padrão do experimento (podem ser sobrescritos por argumentos)
DEFAULT_NUM_AGENTS = 5
DEFAULT_MAX_ITERATIONS = 20

# Variações do experimento
EXPERIMENT_VARIATIONS = {
    "All": ["VND", "ILS", "VNS"],
    "ILS": ["ILS"],
    "VND": ["VND"],
    "VNS": ["VNS"]
}

def agent_worker(agent_id, max_iterations, global_blackboard, instance_path, instance_name, metaheuristics):
    """
    Worker function for each agent process.
    
    Args:
        agent_id: Agent ID
        max_iterations: Maximum number of iterations
        global_blackboard: Shared GlobalBest instance
        instance_path: Path to the instance file
        instance_name: Name of the instance (for logging)
        metaheuristics: List of metaheuristics to use (restricts agent to only these)
    """
    # Inject the shared blackboard into the module
    import src.shared.blackboard
    src.shared.blackboard.global_best = global_blackboard
    
    # Set instance name for logging (use experiment directory to avoid conflicts)
    set_instance_name(f"experiment_{instance_name}")
    
    # Initialize the agent with the specified instance and metaheuristics
    # This restricts the agent to ONLY use the specified metaheuristics
    initialize_agent_with_metaheuristics(agent_id, instance_path, metaheuristics)
    
    # Verify that the agent was initialized with the correct metaheuristics
    from src.main import AGENTS
    agent_beliefs = AGENTS[agent_id]["beliefs"]
    available_actions = list(agent_beliefs.actions.keys())
    if set(available_actions) != set(metaheuristics):
        raise ValueError(
            f"[ERROR] Agent {agent_id} was not initialized correctly. "
            f"Expected: {metaheuristics}, Got: {available_actions}"
        )
    
    # Log available metaheuristics for debugging
    logger = get_logger(agent_id, f"experiment_{instance_name}")
    logger.log(f"[EXPERIMENT] Agent {agent_id} initialized with metaheuristics: {available_actions}")
    
    # Execute the agent cycle
    for iteration in range(max_iterations):
        run_cycle(agent_id)

def initialize_agent_with_metaheuristics(agent_id: str, instance_path: str, metaheuristics: list):
    """Initialize agent with specific metaheuristics"""
    from src.agent_beliefs import AgentBeliefs
    from src.utils.generate_random_feasible_route import generate_random_feasible_route
    from src.utils.evaluator import evaluate_route
    from src.utils.load_instance import load_instance
    
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

def run_experiment_for_instance(instance_path: str, instance_name: str, variation_name: str, metaheuristics: list, num_agents: int, max_iterations: int):
    """
    Run experiment for a specific instance and variation.
    
    Args:
        instance_path: Path to the instance file
        instance_name: Name of the instance
        variation_name: Name of the variation
        metaheuristics: List of metaheuristics to use
        num_agents: Number of agents to run
        max_iterations: Maximum number of iterations per agent
    
    Returns:
        g_best_cost: Best cost found, or None if no solution found
    """
    mp.set_start_method("spawn", force=True)
    
    manager = mp.Manager()
    global_blackboard = GlobalBest(manager)
    
    processes = []
    
    # Create and start processes for each agent
    for i in range(num_agents):
        p = mp.Process(
            target=agent_worker,
            args=(f"agent_{i}", max_iterations, global_blackboard, instance_path, instance_name, metaheuristics)
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

def get_instance_files(instances_dir: str = "instances"):
    """
    Get all instance files from the specified directory.
    
    Note: Only uses instances from 'instances/' directory, not 'big_instances/'.
    """
    instances_path = Path(instances_dir)
    if not instances_path.exists():
        raise FileNotFoundError(f"Directory {instances_dir} not found")
    
    instance_files = []
    for file in sorted(instances_path.glob("*.json")):
        instance_name = file.stem
        instance_files.append((str(file), instance_name))
    
    return instance_files

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comparative experiment between different metaheuristic configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (5 agents, 20 iterations)
  python run_experiment.py
  
  # Run with custom number of agents and iterations
  python run_experiment.py --num-agents 5 --max-iterations 50
        """
    )
    
    parser.add_argument(
        "--num-agents",
        type=int,
        default=DEFAULT_NUM_AGENTS,
        help=f"Number of agents to run (default: {DEFAULT_NUM_AGENTS})"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum number of iterations per agent (default: {DEFAULT_MAX_ITERATIONS})"
    )
    
    args = parser.parse_args()
    
    num_agents = args.num_agents
    max_iterations = args.max_iterations
    
    print("=" * 80)
    print(" EXPERIMENTO COMPARATIVO - METAHEURÍSTICAS")
    print("=" * 80)
    print(f"Número de agentes: {num_agents}")
    print(f"Iterações por agente: {max_iterations}")
    print(f"Variações: {list(EXPERIMENT_VARIATIONS.keys())}")
    print("=" * 80)
    print()
    
    # Get all instances from 'instances/' directory only (not 'big_instances/')
    # Note: Large instances (60, 100 nodes) are in 'big_instances/' and are excluded
    try:
        all_instances = get_instance_files("instances")
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return
    
    if not all_instances:
        print("Nenhuma instância encontrada em 'instances/'!")
        return
    
    print(f"Instâncias encontradas em 'instances/': {len(all_instances)}")
    for path, name in all_instances:
        print(f"  - {name}")
    print()
    
    # Verify that big_instances is not being used
    big_instances_path = Path("big_instances")
    if big_instances_path.exists():
        big_count = len(list(big_instances_path.glob("*.json")))
        if big_count > 0:
            print(f"Nota: {big_count} instância(s) em 'big_instances/' serão ignoradas (instâncias grandes)")
            print()
    
    # Results dictionary: {instance_name: {variation: cost}}
    results = {}
    
    # Run experiments
    total_experiments = len(EXPERIMENT_VARIATIONS) * len(all_instances)
    current = 0
    
    for variation_name, metaheuristics in EXPERIMENT_VARIATIONS.items():
        print(f"\n{'=' * 80}")
        print(f" VARIAÇÃO: {variation_name} - Metaheurísticas: {metaheuristics}")
        print(f"{'=' * 80}\n")
        
        for instance_path, instance_name in all_instances:
            current += 1
            print(f"[{current}/{total_experiments}] Executando {variation_name} para instância {instance_name}...", end=" ", flush=True)
            
            try:
                g_cost = run_experiment_for_instance(
                    instance_path, instance_name, variation_name, metaheuristics, num_agents, max_iterations
                )
                
                if instance_name not in results:
                    results[instance_name] = {}
                
                results[instance_name][variation_name] = g_cost
                
                if g_cost is not None:
                    print(f"✓ Custo: {g_cost}")
                else:
                    print(f"❌ Nenhuma solução encontrada")
                    
            except Exception as e:
                print(f"❌ Erro: {e}")
                if instance_name not in results:
                    results[instance_name] = {}
                results[instance_name][variation_name] = None
    
    # Generate results table
    print("\n" + "=" * 80)
    print(" GERANDO TABELA DE RESULTADOS")
    print("=" * 80)
    
    # Create CSV table
    csv_filename = "experiment_results.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        header = ["Instance"] + list(EXPERIMENT_VARIATIONS.keys())
        writer.writerow(header)
        
        # Data rows
        for instance_name in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            row = [instance_name]
            for variation in EXPERIMENT_VARIATIONS.keys():
                cost = results[instance_name].get(variation)
                if cost is not None:
                    row.append(f"{cost:.2f}")
                else:
                    row.append("N/A")
            writer.writerow(row)
    
    print(f"\n✓ Tabela salva em: {csv_filename}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print(" RESUMO DOS RESULTADOS")
    print("=" * 80)
    print(f"{'Instance':<12}", end="")
    for variation in EXPERIMENT_VARIATIONS.keys():
        print(f"{variation:>15}", end="")
    print()
    print("-" * 80)
    
    for instance_name in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        print(f"{instance_name:<12}", end="")
        for variation in EXPERIMENT_VARIATIONS.keys():
            cost = results[instance_name].get(variation)
            if cost is not None:
                print(f"{cost:>15.2f}", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()
    
    print("\n" + "=" * 80)
    print(" EXPERIMENTO CONCLUÍDO")
    print("=" * 80)

if __name__ == "__main__":
    main()

