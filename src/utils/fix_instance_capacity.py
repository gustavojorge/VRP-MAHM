#!/usr/bin/env python3
"""
Script para ajustar instâncias VRP tornando-as viáveis.

Ajusta n_boardings e n_alighting para que a demanda líquida
seja menor ou igual à capacidade máxima do veículo.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_net_demand(instance: dict) -> Tuple[int, int, int]:
    """
    Calcula a demanda líquida da instância.
    
    Returns:
        (total_boardings, total_alightings, net_demand)
    """
    nodes = instance["nodes"]
    total_boardings = sum(n.get("n_boardings", 0) for n in nodes)
    total_alightings = sum(n.get("n_alighting", 0) for n in nodes)
    net_demand = total_boardings - total_alightings
    
    return total_boardings, total_alightings, net_demand


def adjust_instance_capacity(instance: dict, target_net_demand: int) -> Dict[str, int]:
    """
    Ajusta n_boardings e n_alighting para atingir a demanda líquida alvo.
    
    Args:
        instance: Dicionário da instância
        target_net_demand: Demanda líquida desejada (deve ser <= max_capacity)
    
    Returns:
        Dicionário com estatísticas das mudanças
    """
    nodes = instance["nodes"]
    max_capacity = instance["vehicle_fleet"]["max_capacity"]
    
    # Separar depot e stops
    depot = None
    stops = []
    
    for node in nodes:
        if node.get("type") == "depot":
            depot = node
        else:
            stops.append(node)
    
    # Calcular demanda atual
    current_boardings, current_alightings, current_net = calculate_net_demand(instance)
    
    # Calcular quanto precisa ajustar
    adjustment_needed = current_net - target_net_demand
    
    if adjustment_needed <= 0:
        # Já está viável
        return {
            "nodes_modified": 0,
            "boardings_changed": 0,
            "alightings_changed": 0,
            "total_adjustment": 0
        }
    
    # Estratégia: distribuir o ajuste entre os nós
    # Priorizar aumentar alightings (mais realista)
    # Se necessário, também reduzir boardings
    
    stats = {
        "nodes_modified": 0,
        "boardings_changed": 0,
        "alightings_changed": 0,
        "total_adjustment": 0
    }
    
    # Fase 1: Aumentar alightings para reduzir demanda líquida
    # Distribuir o ajuste proporcionalmente entre os nós
    remaining_adjustment = adjustment_needed
    
    # Ordenar nós por alighting atual (menor primeiro, para dar mais espaço)
    stops_sorted = sorted(stops, key=lambda n: n.get("n_alighting", 0))
    
    for node in stops_sorted:
        if remaining_adjustment <= 0:
            break
        
        current_alighting = node.get("n_alighting", 0)
        current_boarding = node.get("n_boardings", 0)
        
        # Calcular quanto podemos aumentar o alighting
        # Limitar para não exceder o boarding (realismo)
        max_alighting_increase = min(
            remaining_adjustment,
            current_boarding - current_alighting,  # Não exceder boardings
            20  # Limite máximo de aumento por nó
        )
        
        if max_alighting_increase > 0:
            new_alighting = current_alighting + max_alighting_increase
            node["n_alighting"] = new_alighting
            remaining_adjustment -= max_alighting_increase
            
            stats["nodes_modified"] += 1
            stats["alightings_changed"] += max_alighting_increase
            stats["total_adjustment"] += max_alighting_increase
    
    # Fase 2: Se ainda precisar ajustar, reduzir boardings
    if remaining_adjustment > 0:
        # Ordenar por boarding (maior primeiro)
        stops_sorted = sorted(stops, key=lambda n: n.get("n_boardings", 0), reverse=True)
        
        for node in stops_sorted:
            if remaining_adjustment <= 0:
                break
            
            current_boarding = node.get("n_boardings", 0)
            
            # Reduzir boarding, mas manter pelo menos 1
            reduction = min(remaining_adjustment, current_boarding - 1, 15)
            
            if reduction > 0:
                node["n_boardings"] = current_boarding - reduction
                remaining_adjustment -= reduction
                
                if node not in [s for s in stops if s.get("id") == node.get("id")]:
                    stats["nodes_modified"] += 1
                stats["boardings_changed"] += reduction
                stats["total_adjustment"] += reduction
    
    # Fase 3: Se ainda precisar, aumentar mais os alightings (mesmo que exceda boardings)
    if remaining_adjustment > 0:
        stops_sorted = sorted(stops, key=lambda n: n.get("n_alighting", 0))
        
        for node in stops_sorted:
            if remaining_adjustment <= 0:
                break
            
            current_alighting = node.get("n_alighting", 0)
            increase = min(remaining_adjustment, 10)
            
            node["n_alighting"] = current_alighting + increase
            remaining_adjustment -= increase
            
            if node not in [s for s in stops if s.get("id") == node.get("id")]:
                stats["nodes_modified"] += 1
            stats["alightings_changed"] += increase
            stats["total_adjustment"] += increase
    
    return stats


def fix_instance(instance_path: Path, target_ratio: float = 0.9) -> Dict:
    """
    Corrige uma instância para torná-la viável.
    
    Args:
        instance_path: Caminho para o arquivo JSON da instância
        target_ratio: Razão da capacidade a usar como alvo (0.9 = 90% da capacidade)
    
    Returns:
        Dicionário com informações sobre as mudanças
    """
    # Ler instância
    with open(instance_path, 'r', encoding='utf-8') as f:
        instance = json.load(f)
    
    max_capacity = instance["vehicle_fleet"]["max_capacity"]
    target_net_demand = int(max_capacity * target_ratio)
    
    # Calcular demanda atual
    boardings_before, alightings_before, net_before = calculate_net_demand(instance)
    
    result = {
        "file": str(instance_path),
        "instance_id": instance.get("instance_id", "unknown"),
        "num_nodes": instance.get("num_nodes", 0),
        "max_capacity": max_capacity,
        "net_demand_before": net_before,
        "net_demand_after": None,
        "target_net_demand": target_net_demand,
        "was_viable": net_before <= max_capacity,
        "is_viable_after": None,
        "modified": False,
        "stats": {}
    }
    
    # Se já está viável, não precisa modificar
    if net_before <= max_capacity:
        result["net_demand_after"] = net_before
        result["is_viable_after"] = True
        return result
    
    # Ajustar
    stats = adjust_instance_capacity(instance, target_net_demand)
    
    # Recalcular demanda após ajuste
    boardings_after, alightings_after, net_after = calculate_net_demand(instance)
    
    # Salvar instância modificada
    with open(instance_path, 'w', encoding='utf-8') as f:
        json.dump(instance, f, indent=2, ensure_ascii=False)
    
    result["net_demand_after"] = net_after
    result["is_viable_after"] = net_after <= max_capacity
    result["modified"] = True
    result["stats"] = stats
    result["boardings_before"] = boardings_before
    result["boardings_after"] = boardings_after
    result["alightings_before"] = alightings_before
    result["alightings_after"] = alightings_after
    
    return result


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ajusta instâncias VRP para torná-las viáveis (demanda líquida <= capacidade)"
    )
    parser.add_argument(
        "--instances-dir",
        type=str,
        default="instances",
        help="Diretório contendo as instâncias (default: instances)"
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.9,
        help="Razão da capacidade a usar como alvo (0.0-1.0, default: 0.9 = 90%%)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas simular, não salvar alterações"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="capacity_fix_report.txt",
        help="Arquivo de relatório (default: capacity_fix_report.txt)"
    )
    
    args = parser.parse_args()
    
    instances_dir = Path(args.instances_dir)
    if not instances_dir.exists():
        print(f"❌ Diretório não encontrado: {instances_dir}")
        return
    
    # Encontrar todos os arquivos JSON (exceto index.json)
    instance_files = [
        f for f in instances_dir.glob("*.json")
        if f.name != "index.json"
    ]
    
    if not instance_files:
        print(f"❌ Nenhuma instância encontrada em {instances_dir}")
        return
    
    print("=" * 80)
    print(" AJUSTE DE CAPACIDADE DAS INSTÂNCIAS VRP")
    print("=" * 80)
    print(f"Diretório: {instances_dir}")
    print(f"Instâncias encontradas: {len(instance_files)}")
    print(f"Razão alvo: {args.target_ratio * 100:.1f}% da capacidade")
    print(f"Modo: {'DRY RUN (simulação)' if args.dry_run else 'EXECUÇÃO REAL'}")
    print("=" * 80)
    print()
    
    results = []
    modified_count = 0
    already_viable_count = 0
    
    for instance_file in sorted(instance_files):
        print(f"Processando: {instance_file.name}...", end=" ", flush=True)
        
        try:
            if args.dry_run:
                # Em dry-run, apenas calcular sem modificar
                with open(instance_file, 'r', encoding='utf-8') as f:
                    instance = json.load(f)
                
                max_capacity = instance["vehicle_fleet"]["max_capacity"]
                _, _, net_demand = calculate_net_demand(instance)
                target_net = int(max_capacity * args.target_ratio)
                
                result = {
                    "file": str(instance_file),
                    "instance_id": instance.get("instance_id", "unknown"),
                    "num_nodes": instance.get("num_nodes", 0),
                    "max_capacity": max_capacity,
                    "net_demand_before": net_demand,
                    "net_demand_after": net_demand if net_demand <= max_capacity else target_net,
                    "target_net_demand": target_net,
                    "was_viable": net_demand <= max_capacity,
                    "is_viable_after": True,
                    "modified": not (net_demand <= max_capacity),
                    "stats": {"nodes_modified": 0} if net_demand <= max_capacity else {"nodes_modified": "estimado"}
                }
            else:
                result = fix_instance(instance_file, args.target_ratio)
            
            results.append(result)
            
            if result["was_viable"]:
                print(f"✓ Já viável (demanda líquida: {result['net_demand_before']})")
                already_viable_count += 1
            elif result["modified"]:
                print(f"✓ Modificado: {result['net_demand_before']} → {result['net_demand_after']} "
                      f"({result['stats'].get('nodes_modified', 0)} nós)")
                modified_count += 1
            else:
                print(f"⚠️  Não modificado (erro?)")
        
        except Exception as e:
            print(f"❌ Erro: {e}")
            results.append({
                "file": str(instance_file),
                "error": str(e)
            })
    
    print()
    print("=" * 80)
    print(" RESUMO")
    print("=" * 80)
    print(f"Total de instâncias: {len(instance_files)}")
    print(f"Já viáveis: {already_viable_count}")
    print(f"Modificadas: {modified_count}")
    print(f"Com erro: {len([r for r in results if 'error' in r])}")
    print()
    
    # Gerar relatório detalhado
    if not args.dry_run:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" RELATÓRIO DE AJUSTE DE CAPACIDADE\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Data: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Diretório: {instances_dir}\n")
            f.write(f"Razão alvo: {args.target_ratio * 100:.1f}% da capacidade\n")
            f.write(f"Total de instâncias: {len(instance_files)}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write(" INSTÂNCIAS MODIFICADAS\n")
            f.write("-" * 80 + "\n\n")
            
            modified_results = [r for r in results if r.get("modified", False)]
            if modified_results:
                for r in modified_results:
                    f.write(f"Arquivo: {Path(r['file']).name}\n")
                    f.write(f"  Instance ID: {r.get('instance_id', 'N/A')}\n")
                    f.write(f"  Nós: {r.get('num_nodes', 'N/A')}\n")
                    f.write(f"  Capacidade máxima: {r.get('max_capacity', 'N/A')}\n")
                    f.write(f"  Demanda líquida: {r['net_demand_before']} → {r['net_demand_after']}\n")
                    f.write(f"  Embarques: {r.get('boardings_before', 'N/A')} → {r.get('boardings_after', 'N/A')}\n")
                    f.write(f"  Desembarques: {r.get('alightings_before', 'N/A')} → {r.get('alightings_after', 'N/A')}\n")
                    stats = r.get('stats', {})
                    f.write(f"  Nós modificados: {stats.get('nodes_modified', 0)}\n")
                    f.write(f"  Ajuste total: {stats.get('total_adjustment', 0)}\n")
                    f.write("\n")
            else:
                f.write("Nenhuma instância foi modificada.\n\n")
            
            f.write("-" * 80 + "\n")
            f.write(" INSTÂNCIAS JÁ VIÁVEIS (não modificadas)\n")
            f.write("-" * 80 + "\n\n")
            
            viable_results = [r for r in results if r.get("was_viable", False) and not r.get("modified", False)]
            if viable_results:
                for r in viable_results:
                    f.write(f"{Path(r['file']).name}: demanda líquida = {r['net_demand_before']} "
                           f"(capacidade = {r.get('max_capacity', 'N/A')})\n")
            else:
                f.write("Nenhuma instância já estava viável.\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ Relatório salvo em: {args.report}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

