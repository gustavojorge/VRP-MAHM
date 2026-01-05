#!/usr/bin/env python3
"""
Script to validate VRP instance structure.
"""
import json
from pathlib import Path

def validate_instance(instance_path: str):
    """Validate a VRP instance file."""
    instance_path_obj = Path(instance_path)
    
    if not instance_path_obj.exists():
        print(f'❌ Arquivo não encontrado: {instance_path}')
        return False
    
    try:
        with open(instance_path_obj, 'r') as f:
            instance = json.load(f)
    except json.JSONDecodeError as e:
        print(f'❌ Erro ao ler JSON: {e}')
        return False
    
    print(f'\n{"="*60}')
    print(f'Validando: {instance_path}')
    print(f'{"="*60}')
    
    # Verificar campos obrigatórios
    required_fields = ['num_nodes', 'nodes', 'trip_time_matrix', 'vehicle_fleet']
    missing_fields = [f for f in required_fields if f not in instance]
    if missing_fields:
        print(f'❌ Campos faltando: {missing_fields}')
        return False
    
    # Informações básicas
    num_nodes = instance['num_nodes']
    nodes = instance['nodes']
    matrix = instance['trip_time_matrix']
    max_capacity = instance['vehicle_fleet']['max_capacity']
    
    print(f'\n📊 Informações Básicas:')
    print(f'  num_nodes (declarado): {num_nodes}')
    print(f'  Total de nós no array: {len(nodes)}')
    print(f'  Capacidade máxima: {max_capacity}')
    
    # Verificar consistência do número de nós
    if len(nodes) != num_nodes:
        print(f'⚠️  AVISO: num_nodes ({num_nodes}) != len(nodes) ({len(nodes)})')
    
    # Verificar IDs dos nós
    node_ids = [n['id'] for n in nodes]
    expected_ids = list(range(len(nodes)))
    
    print(f'\n🔢 Validação de IDs:')
    print(f'  IDs encontrados: {min(node_ids)} a {max(node_ids)}')
    print(f'  IDs esperados: {min(expected_ids)} a {max(expected_ids)}')
    
    if set(node_ids) != set(expected_ids):
        missing_ids = set(expected_ids) - set(node_ids)
        extra_ids = set(node_ids) - set(expected_ids)
        if missing_ids:
            print(f'  ❌ IDs faltando: {sorted(missing_ids)}')
        if extra_ids:
            print(f'  ❌ IDs extras: {sorted(extra_ids)}')
        return False
    else:
        print(f'  ✓ IDs estão corretos')
    
    # Verificar se há depot (id 0)
    depot = [n for n in nodes if n['id'] == 0]
    if not depot:
        print(f'  ❌ Depot (id=0) não encontrado!')
        return False
    else:
        print(f'  ✓ Depot encontrado: {depot[0].get("name", "N/A")}')
    
    # Verificar matriz de distâncias
    print(f'\n📐 Validação da Matriz:')
    print(f'  Tamanho da matriz: {len(matrix)}x{len(matrix[0]) if matrix else 0}')
    print(f'  Tamanho esperado: {len(nodes)}x{len(nodes)}')
    
    if len(matrix) != len(nodes):
        print(f'  ❌ Número de linhas incorreto! Esperado {len(nodes)}, encontrado {len(matrix)}')
        return False
    
    # Verificar se a matriz é quadrada
    for i, row in enumerate(matrix):
        if len(row) != len(nodes):
            print(f'  ❌ Linha {i} tem {len(row)} colunas, esperado {len(nodes)}')
            return False
    
    print(f'  ✓ Matriz é quadrada e consistente')
    
    # Verificar diagonal (deve ser zero)
    diagonal_ok = True
    for i in range(len(matrix)):
        if matrix[i][i] != 0:
            print(f'  ⚠️  AVISO: Diagonal[{i}][{i}] = {matrix[i][i]} (deveria ser 0)')
            diagonal_ok = False
    
    if diagonal_ok:
        print(f'  ✓ Diagonal contém apenas zeros')
    
    # Verificar capacidade
    print(f'\n🚌 Validação de Capacidade:')
    total_boardings = sum(n.get('n_boardings', 0) for n in nodes)
    total_alightings = sum(n.get('n_alighting', 0) for n in nodes)
    net_demand = total_boardings - total_alightings
    
    print(f'  Total de embarques: {total_boardings}')
    print(f'  Total de desembarques: {total_alightings}')
    print(f'  Demanda líquida: {net_demand}')
    print(f'  Capacidade máxima: {max_capacity}')
    
    if net_demand > max_capacity:
        print(f'  ⚠️  AVISO: Demanda líquida ({net_demand}) > capacidade ({max_capacity})')
        print(f'     Isso pode tornar difícil gerar rotas viáveis!')
    else:
        print(f'  ✓ Demanda líquida está dentro da capacidade')
    
    # Verificar se todos os nós têm os campos necessários
    print(f'\n📋 Validação de Campos dos Nós:')
    required_node_fields = ['id', 'type', 'n_boardings', 'n_alighting']
    missing_in_nodes = []
    
    for node in nodes:
        for field in required_node_fields:
            if field not in node:
                missing_in_nodes.append(f"Node {node.get('id', '?')}: campo '{field}' faltando")
    
    if missing_in_nodes:
        print(f'  ❌ Campos faltando em alguns nós:')
        for msg in missing_in_nodes[:5]:  # Mostrar apenas os primeiros 5
            print(f'    - {msg}')
        if len(missing_in_nodes) > 5:
            print(f'    ... e mais {len(missing_in_nodes) - 5} nós')
        return False
    else:
        print(f'  ✓ Todos os nós têm os campos necessários')
    
    print(f'\n{"="*60}')
    print(f'✅ Instância válida!')
    print(f'{"="*60}\n')
    
    return True

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        instance_path = sys.argv[1]
    else:
        instance_path = 'instances/100.json'
    
    success = validate_instance(instance_path)
    sys.exit(0 if success else 1)

