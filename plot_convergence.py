"""
Script para gerar gráfico de convergência do g_best_cost em função do budget
(checkpoint_evaluations) para a instância 100, comparando as 4 ações (MAHM, ILS, VND, VNS).
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_convergence(instance_id=100, run_id=10, min_budget=20000, max_budget=200000):
    """
    Gera gráfico de convergência comparando as 4 ações.
    
    Args:
        instance_id: ID da instância (padrão: 100)
        run_id: ID do run a ser analisado (padrão: 10)
        min_budget: Budget mínimo para filtrar (padrão: 20000)
        max_budget: Budget máximo para filtrar (padrão: 200000)
    """
    base_path = Path(f"logs/{instance_id}")
    actions = ["mahm", "ils", "vnd", "vns"]
    
    # Mapeamento de ações para labels de exibição
    action_labels = {
        "mahm": "MAHM",
        "ils": "ILS",
        "vnd": "VND",
        "vns": "VNS"
    }
    
    # Cores para cada ação
    colors = {
        "mahm": "#1f77b4",  # Azul
        "ils": "#ff7f0e",   # Laranja
        "vnd": "#2ca02c",   # Verde
        "vns": "#d62728"    # Vermelho
    }
    
    plt.figure(figsize=(12, 7))
    
    for action in actions:
        csv_path = base_path / action / str(run_id) / "checkpoint.csv"
        
        if not csv_path.exists():
            print(f"⚠️  Arquivo não encontrado: {csv_path}")
            continue
        
        try:
            # Ler CSV
            df = pd.read_csv(csv_path)
            
            # Filtrar dados pelo range de budget
            df_filtered = df[
                (df["checkpoint_evaluations"] >= min_budget) &
                (df["checkpoint_evaluations"] <= max_budget)
            ]
            
            if df_filtered.empty:
                print(f"⚠️  Nenhum dado encontrado no range para {action}")
                continue
            
            # Extrair dados para plotagem
            x = df_filtered["checkpoint_evaluations"]
            y = df_filtered["g_best_cost"]
            
            # Plotar linha
            label = action_labels.get(action, action.upper())
            color = colors.get(action, None)
            plt.plot(x, y, marker='o', label=label, linewidth=2, markersize=6, color=color)
            
        except Exception as e:
            print(f"❌ Erro ao processar {action}: {e}")
            continue
    
    # Configurar gráfico
    plt.xlabel("Budget (evaluations)", fontsize=16, fontweight='bold')
    plt.ylabel("G_Best Cost", fontsize=16, fontweight='bold')
    plt.title(f"G_Best Convergence - Instance {instance_id}", fontsize=18, fontweight='bold')
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Formatar eixo X para melhor legibilidade
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar gráfico
    output_file = f"convergence_instance_{instance_id}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo em: {output_file}")
    
    # Fechar figura para liberar memória
    plt.close()


if __name__ == "__main__":
    plot_convergence()

