from agents.utils.load_instance import load_instance
from agents.utils.generate_random_feasible_route import generate_random_feasible_route
from agents.problem.evaluator import evaluate_route
from agents.methods.path_relinking import path_relinking
from agents.actions.vnd.vnd import vnd


def verbose_intensification(route, instance):
    """
    Wrapper para VND com prints (intensificação)
    """
    print("\n>>> INTENSIFICAÇÃO DISPARADA (VND)")
    print("Rota base:", route)

    new_route, new_cost = vnd(route, instance)

    print("Rota após intensificação:", new_route)
    print("Custo após intensificação:", new_cost)
    print(">>> FIM DA INTENSIFICAÇÃO\n")

    return new_route, new_cost


def main():
    instance = load_instance("instances/1.json")

    # Solução de origem (Ps)
    origin = generate_random_feasible_route(instance)

    # Solução alvo (Pt) — outra solução aleatória para teste
    target = generate_random_feasible_route(instance)

    print("\n==============================")
    print("TESTE PATH-RELINKING")
    print("==============================")

    print("\nPs (origem):", origin)
    feasible, cost = evaluate_route(origin, instance)
    print(f"Viável: {feasible} | Custo: {cost}")

    print("\nPt (target):", target)
    feasible, cost = evaluate_route(target, instance)
    print(f"Viável: {feasible} | Custo: {cost}")

    print("\n--- Iniciando Path-Relinking ---\n")

    best_route, best_cost = path_relinking(
        origin=origin,
        target=target,
        instance=instance,
        intensification_method=verbose_intensification
    )

    print("\n==============================")
    print("RESULTADO FINAL")
    print("==============================")
    print("Rota retornada:", best_route)
    print("Custo retornado:", best_cost)

    feasible, _ = evaluate_route(best_route, instance)
    print("Viável:", feasible)


if __name__ == "__main__":
    main()
