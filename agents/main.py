import asyncio
from spade_bdi.bdi import BDIAgent
from agents.utils.load_instance import load_instance
from agents.utils.generate_random_feasible_route import generate_random_feasible_route
from agents.utils.compute_route_cost import compute_route_cost

async def main():
    instance_path = "instances/1.json"
    instance = load_instance(instance_path)

    # Geração da solução inicial (pbest)
    route = generate_random_feasible_route(instance)
    cost = compute_route_cost(route, instance["trip_time_matrix"])

    print("=== SOLUÇÃO INICIAL GERADA ===")
    print("Rota:", route)
    print("Custo:", cost)

    # Criação do agente BDI
    agent = BDIAgent(
        "agent1@localhost",
        "123",
        "agents/agent.asl"
    )

    await agent.start()

    # Armazenando crenças iniciais
    agent.bdi.set_belief("pbest_route", *route)
    agent.bdi.set_belief("pbest_cost", cost)

    print("\n=== CRENÇAS DO AGENTE ===")
    agent.bdi.print_beliefs()

    await asyncio.sleep(2)
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
