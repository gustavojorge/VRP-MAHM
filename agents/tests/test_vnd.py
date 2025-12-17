from agents.actions.vnd.vnd import vnd
from agents.problem.evaluator import evaluate_route
from agents.utils.load_instance import load_instance


def test_vnd_basic():
    print("\n==============================")
    print("TESTE DO VND")
    print("==============================")

    # 🔹 Carrega instância real
    instance = load_instance("instances/1.json")

    # Gera rota inicial viável
    initial_route = [0, 2, 1, 3, 4, 0]

    feasible, initial_cost = evaluate_route(initial_route, instance)

    print(f"Rota inicial: {initial_route}")
    print(f"Viável: {feasible}")
    print(f"Custo inicial: {initial_cost}")

    if not feasible:
        print("❌ Rota inicial inviável. Abortando teste.")
        return

    print("\n>>> DISPARANDO VND")

    final_route, final_cost = vnd(initial_route, instance)

    print("\n>>> RESULTADO DO VND")
    print(f"Rota final: {final_route}")
    print(f"Custo final: {final_cost}")

    if final_cost < initial_cost:
        print("✔ VND encontrou melhoria")
    elif final_cost == initial_cost:
        print("➖ VND não melhorou a solução")
    else:
        print("⚠ VND piorou a solução (verificar implementação)")

    print("\n==============================")
    print("FIM DO TESTE DO VND")
    print("==============================")


if __name__ == "__main__":
    test_vnd_basic()
