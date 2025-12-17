from agents.main import initialize_agent, run_cycle, AGENTS
from agents.shared.blackboard import global_best

MAX_ITERATIONS = 50
AGENT_ID = "agent1"


def main():
    print("===================================")
    print(" INICIANDO EXECUÇÃO DE UM AGENTE ")
    print("===================================")

    # Inicializa agente
    initialize_agent(AGENT_ID)

    beliefs = AGENTS[AGENT_ID]["beliefs"]

    print("\n--- Estado inicial ---")
    print(f"Rota inicial: {beliefs.current_route}")
    print(f"Custo inicial: {beliefs.current_cost}")
    print("----------------------\n")

    # Loop principal (simula o Jason)
    for it in range(MAX_ITERATIONS):
        print(f"\n>>> ITERAÇÃO {it}")

        run_cycle(AGENT_ID)

        print(f"Custo atual : {beliefs.current_cost}")
        print(f"p_best      : {beliefs.p_best_cost}")

        g_route, g_cost, g_agent = global_best.get()
        if g_route is not None:
            print(f"g_best      : {g_cost} (agente {g_agent})")

    print("\n===================================")
    print(" EXECUÇÃO FINALIZADA ")
    print("===================================")

    print("\n--- Resultado final ---")
    print(f"p_best do agente : {beliefs.p_best_cost}")
    print(f"Rota p_best     : {beliefs.p_best_route}")

    g_route, g_cost, g_agent = global_best.get()
    print(f"\nGlobal Best:")
    print(f"Custo : {g_cost}")
    print(f"Rota  : {g_route}")
    print(f"Agente: {g_agent}")


if __name__ == "__main__":
    main()
