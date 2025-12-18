#!/usr/bin/env python3
"""
Script para executar o sistema multi-agente para todas as instâncias.

Percorre todas as instâncias em instances/ e executa run_multi_agent para cada uma.
Os logs são organizados por instância em logs/{instance_name}/
"""

import os
import argparse
from pathlib import Path
from run_multi_agent import run_for_instance

def get_instance_files(instances_dir: str = "instances"):
    """
    Retorna lista de arquivos de instância.
    
    Returns:
        Lista de tuplas (caminho_completo, nome_instancia)
    """
    instances_path = Path(instances_dir)
    if not instances_path.exists():
        raise FileNotFoundError(f"Directory {instances_dir} not found")
    
    instance_files = []
    for file in sorted(instances_path.glob("*.json")):
        # Nome da instância é o nome do arquivo sem extensão
        instance_name = file.stem
        instance_files.append((str(file), instance_name))
    
    return instance_files

def main():
    parser = argparse.ArgumentParser(
        description="Run multi-agent system for all instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (2 agents, 10 iterations)
  python run_all_instances.py
  
  # Run with custom number of agents and iterations
  python run_all_instances.py --num-agents 4 --max-iterations 20
  
  # Run only specific instance
  python run_all_instances.py --instance 1
        """
    )
    
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Number of agents to run (default: 2)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of iterations per agent (default: 10)"
    )
    
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Run only for a specific instance (e.g., '1' or '50'). If not specified, runs for all instances."
    )
    
    parser.add_argument(
        "--instances-dir",
        type=str,
        default="instances",
        help="Directory containing instance files (default: 'instances')"
    )
    
    args = parser.parse_args()
    
    # Get instance files
    try:
        all_instances = get_instance_files(args.instances_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if not all_instances:
        print(f"No instance files found in {args.instances_dir}/")
        return
    
    # Filter by instance if specified
    if args.instance:
        all_instances = [
            (path, name) for path, name in all_instances 
            if name == args.instance
        ]
        if not all_instances:
            print(f"Instance '{args.instance}' not found in {args.instances_dir}/")
            return
    
    print("=" * 60)
    print(" MULTI-AGENT SYSTEM - RUNNING FOR ALL INSTANCES")
    print("=" * 60)
    print(f"Number of agents: {args.num_agents}")
    print(f"Max iterations per agent: {args.max_iterations}")
    print(f"Instances to process: {len(all_instances)}")
    print("=" * 60)
    print()
    
    # Process each instance
    for idx, (instance_path, instance_name) in enumerate(all_instances, 1):
        print(f"\n{'=' * 60}")
        print(f" Processing Instance {idx}/{len(all_instances)}: {instance_name}")
        print(f" File: {instance_path}")
        print(f"{'=' * 60}\n")
        
        try:
            run_for_instance(
                instance_path=instance_path,
                instance_name=instance_name,
                num_agents=args.num_agents,
                max_iterations=args.max_iterations
            )
            print(f"\n✓ Instance {instance_name} completed successfully")
        except Exception as e:
            print(f"\n❌ Error processing instance {instance_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(" ALL INSTANCES PROCESSED")
    print("=" * 60)
    print(f"\nLogs are organized in: logs/{{instance_name}}/agent_X.log")
    print(f"Example: logs/1/agent_0.log, logs/50/agent_0.log, etc.")

if __name__ == "__main__":
    main()

