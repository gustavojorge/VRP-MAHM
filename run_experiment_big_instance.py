#!/usr/bin/env python3
"""
Script para executar experimento em uma instância grande de big_instances/
e anexar os resultados ao experiment_results.csv existente.
"""

import multiprocessing as mp
import csv
from pathlib import Path
from src.shared.blackboard import GlobalBest
from src.main import AGENTS
from src.utils.logger import set_instance_name, get_logger
# Import from run_experiment
import sys
sys.path.insert(0, '.')

# Import constants and functions
from run_experiment import (
    EXPERIMENT_VARIATIONS,
    DEFAULT_NUM_AGENTS,
    DEFAULT_MAX_ITERATIONS,
    initialize_agent_with_metaheuristics
)

# Import agent_worker from run_experiment
from run_experiment import agent_worker

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

def append_to_csv(instance_name: str, results: dict, csv_filename: str = "experiment_results.csv"):
    """
    Append results for a single instance to the existing CSV file.
    If the instance already exists, it will be updated.
    """
    # Read existing CSV if it exists
    existing_data = {}
    if Path(csv_filename).exists():
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_data[row['Instance']] = row
    
    # Update or add the new instance results
    row_data = {'Instance': instance_name}
    for var in EXPERIMENT_VARIATIONS.keys():
        cost = results.get(var)
        if cost is not None:
            row_data[var] = f"{cost:.2f}"
        else:
            row_data[var] = "N/A"
    existing_data[instance_name] = row_data
    
    # Write back to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Instance'] + list(EXPERIMENT_VARIATIONS.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort instances numerically
        sorted_instances = sorted(existing_data.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        for inst in sorted_instances:
            writer.writerow(existing_data[inst])
    
    print(f"\n✓ Resultados anexados/atualizados em: {csv_filename}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run experiment for a big instance and append to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (5 agents, 15 iterations)
  python run_experiment_big_instance.py --instance 100
  
  # Run with custom number of agents and iterations
  python run_experiment_big_instance.py --instance 100 --num-agents 4 --max-iterations 20
        """
    )
    
    parser.add_argument(
        "--instance",
        type=str,
        required=True,
        help="Instance name (e.g., '100', '101', '102')"
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
        default=15,
        help="Maximum number of iterations per agent (default: 15 for big instances)"
    )
    
    args = parser.parse_args()
    instance_name = args.instance
    instance_path = f"big_instances/{instance_name}.json"
    num_agents = args.num_agents
    max_iterations = args.max_iterations
    
    # Verify instance exists
    if not Path(instance_path).exists():
        print(f"❌ Erro: Instância {instance_path} não encontrada!")
        return
    
    print("=" * 80)
    print(f" EXPERIMENTO PARA INSTÂNCIA GRANDE: {instance_name}")
    print("=" * 80)
    print(f"Número de agentes: {num_agents}")
    print(f"Iterações por agente: {max_iterations}")
    print(f"Variações: {list(EXPERIMENT_VARIATIONS.keys())}")
    print(f"Arquivo: {instance_path}")
    print("=" * 80)
    print()
    
    # Results dictionary for this instance
    results = {}
    
    # Run experiments for all variations
    total_experiments = len(EXPERIMENT_VARIATIONS)
    current = 0
    
    for variation_name, metaheuristics in EXPERIMENT_VARIATIONS.items():
        current += 1
        print(f"[{current}/{total_experiments}] Executando {variation_name} para instância {instance_name}...", end=" ", flush=True)
        
        try:
            g_cost = run_experiment_for_instance(
                instance_path, instance_name, variation_name, metaheuristics, num_agents, max_iterations
            )
            
            results[variation_name] = g_cost
            
            if g_cost is not None:
                print(f"✓ Custo: {g_cost}")
            else:
                print(f"❌ Nenhuma solução encontrada")
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            results[variation_name] = None
    
    # Append to CSV
    print("\n" + "=" * 80)
    print(" ANEXANDO RESULTADOS AO CSV")
    print("=" * 80)
    
    append_to_csv(instance_name, results)
    
    # Print summary
    print("\n" + "=" * 80)
    print(" RESUMO DOS RESULTADOS")
    print("=" * 80)
    print(f"{'Variation':<15} {'Cost':>15}")
    print("-" * 80)
    
    for variation in EXPERIMENT_VARIATIONS.keys():
        cost = results.get(variation)
        if cost is not None:
            print(f"{variation:<15} {cost:>15.2f}")
        else:
            print(f"{variation:<15} {'N/A':>15}")
    
    print("\n" + "=" * 80)
    print(" EXPERIMENTO CONCLUÍDO")
    print("=" * 80)

if __name__ == "__main__":
    main()

