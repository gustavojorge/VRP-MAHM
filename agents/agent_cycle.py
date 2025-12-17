from typing import Dict, Callable, Tuple, List

from agents.agent_beliefs import AgentBeliefs
from agents.shared.blackboard import global_best
from agents.methods.decision_method import decision_method

from agents.methods.path_relinking import path_relinking
from agents.problem.evaluator import evaluate_route

# Metaheurísticas disponíveis
from agents.actions.vnd.vnd import vnd
from agents.actions.ils.ils import ils


# ---------------------------------------------------------------------
# Registro das metaheurísticas (Ações)
# ---------------------------------------------------------------------

METAHEURISTICS: Dict[str, Callable[[List[int], dict], Tuple[List[int], float]]] = {
    "VND": vnd,
    "ILS": ils,
}


# ---------------------------------------------------------------------
# Ciclo Cognitivo do Agente
# ---------------------------------------------------------------------

def run_agent_cycle(
    beliefs: AgentBeliefs,
    instance: dict
) -> Tuple[List[int], float]:
    """
    Executa UMA iteração completa do agente, seguindo o ciclo:

    1. Decision Method
    2. Execução da Metaheurística
    3. Learning Method
    4. Velocity Operator (Path-Relinking + Intensificação Oportunista)
    5. Atualização do p_best
    6. Atualização do g_best

    Retorna a nova posição final do agente.
    """

    # ------------------------------------------------------------
    # Estado inicial
    # ------------------------------------------------------------
    current_route = beliefs.current_route
    current_cost = beliefs.current_cost

    if current_route is None:
        raise ValueError("Agente sem solução inicial definida")

    # ------------------------------------------------------------
    # 1️⃣ Decision Method
    # ------------------------------------------------------------
    print("1 - Decision Method.")
    action_name = decision_method(beliefs)
    print("Metaheurística a ser executada: ", action_name)
    action_fn = METAHEURISTICS[action_name]

    # ------------------------------------------------------------
    # 2️⃣ Execução da Metaheurística (Ação)
    # ------------------------------------------------------------
    print("2 - Execução da Metaheurística (Ação)")
    new_route, new_cost = action_fn(current_route, instance)

    # ------------------------------------------------------------
    # 3️⃣ Learning Method
    # ------------------------------------------------------------
    print("3 - Learning Method")
    beliefs.update_after_action(
        action_name=action_name,
        old_cost=current_cost,
        new_cost=new_cost
    )

    beliefs.update_current_solution(new_route, new_cost)

    # ------------------------------------------------------------
    # 4️⃣ Velocity Operator (Path-Relinking)
    # ------------------------------------------------------------
    print("4 - Velocity Operator (Path-Relinking)")
    g_best_route, g_best_cost, _ = global_best.get()

    # Se ainda não existe g_best, não há diversificação
    if g_best_route is not None:

        def opportunistic_intensification(route, instance):
            """
            Intensificação oportunista:
            escolhe a melhor metaheurística segundo as crenças atuais.
            """
            best_action = beliefs.get_best_action()
            return METAHEURISTICS[best_action](route, instance)

        final_route, final_cost = path_relinking(
            origin=new_route,
            target=g_best_route,
            instance=instance,
            intensification_method=opportunistic_intensification
        )
    else:
        final_route, final_cost = new_route, new_cost

    beliefs.update_current_solution(final_route, final_cost)

    # ------------------------------------------------------------
    # 5️⃣ Atualização do p_best
    # ------------------------------------------------------------
    print("5 - Atualização do p_best")
    beliefs.try_update_pbest(final_route, final_cost)

    # ------------------------------------------------------------
    # 6️⃣ Atualização do g_best (Blackboard)
    # ------------------------------------------------------------
    print("6 - Atualização do g_best (Blackboard)")
    global_best.try_update(
        candidate_route=final_route,
        candidate_cost=final_cost,
        agent_id=beliefs.agent_id
    )

    return final_route, final_cost
