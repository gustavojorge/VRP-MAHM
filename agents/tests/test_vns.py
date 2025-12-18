from agents.actions.vns import vns
from agents.problem.evaluator import evaluate_route
from agents.utils.load_instance import load_instance


def test_vns_basic():
    print("==============================")
    print("TESTE DO VNS")
    print("==============================")

    instance = load_instance("instances/1.json")

    initial_route = [0, 2, 1, 3, 4, 0]

    feasible, initial_cost = evaluate_route(initial_route, instance)

    print(f"Rota inicial: {initial_route}")
    print(f"Viável: {feasible}")
    print(f"Custo inicial: {initial_cost}")

    if not feasible:
        print("❌ Rota inicial inviável. Abortando teste.")
        return

    print("\n>>> DISPARANDO VNS")

    final_route, final_cost = vns(initial_route, instance, max_iterations=20, seed=42)

    print("\n>>> RESULTADO DO VNS")
    print(f"Rota final: {final_route}")
    print(f"Custo final: {final_cost}")

    if final_cost < initial_cost:
        print("✔ VNS encontrou melhoria")
    elif final_cost == initial_cost:
        print("➖ VNS não melhorou a solução")
    else:
        print("⚠ VNS piorou a solução (verificar implementação)")

    print("\n==============================")
    print("FIM DO TESTE DO VNS")
    print("==============================")


if __name__ == "__main__":
    test_vns_basic()
