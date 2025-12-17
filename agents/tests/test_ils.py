from agents.actions.ils.ils import ils
from agents.problem.evaluator import evaluate_route
from agents.utils.load_instance import load_instance
from agents.utils.generate_random_feasible_route import generate_random_feasible_route


def test_ils_basic():
    print("\n==============================")
    print("TESTE DO ILS")
    print("==============================")

    # 🔹 Instância real
    instance = load_instance("instances/1.json")

    # 🔹 Rota inicial viável
    initial_route = generate_random_feasible_route(instance)

    feasible, initial_cost = evaluate_route(initial_route, instance)

    print("Rota inicial:", initial_route)
    print("Viável:", feasible)
    print("Custo inicial:", initial_cost)

    if not feasible:
        print("❌ Rota inicial inviável. Abortando teste.")
        return

    print("\n>>> DISPARANDO ILS")

    best_route, best_cost = ils(
        initial_route=initial_route,
        instance=instance,
        max_iterations=20
    )

    print("\n==============================")
    print("RESULTADO FINAL DO ILS")
    print("==============================")
    print("Rota final:", best_route)
    print("Custo final:", best_cost)

    feasible, _ = evaluate_route(best_route, instance)
    print("Viável:", feasible)

    if best_cost < initial_cost:
        print("✔ ILS encontrou melhoria")
    elif best_cost == initial_cost:
        print("➖ ILS manteve a solução")
    else:
        print("⚠ ILS piorou a solução (verificar lógica)")

    print("\n==============================")
    print("FIM DO TESTE DO ILS")
    print("==============================")


if __name__ == "__main__":
    test_ils_basic()
