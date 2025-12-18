from typing import Dict, Callable, Tuple, List

from agents.agent_beliefs import AgentBeliefs
from agents.shared.blackboard import global_best
from agents.methods.decision_method import decision_method

from agents.methods.path_relinking import path_relinking
from agents.problem.evaluator import evaluate_route

# Available Metaheuristics
from agents.actions.vnd.vnd import vnd
from agents.actions.vns.vns import vns
from agents.actions.ils.ils import ils

# Metaheuristics (Actions) registry
METAHEURISTICS: Dict[str, Callable[[List[int], dict], Tuple[List[int], float]]] = {
    "VND": vnd,
    "VNS": vns,
    "ILS": ils,
}

# ---------------------------------------------------------------------
# Agent Cognitive Cycle
# ---------------------------------------------------------------------

def run_agent_cycle(
    beliefs: AgentBeliefs,
    instance: dict
) -> Tuple[List[int], float]:
    """
    Runs a complete iteration of the agent, following the cycle:

    1. Decision Method
    2. Execution of the Metaheuristic
    3. Learning Method
    4. Velocity Operator (Path-Relinking)
    5. Update of p_best
    6. Update of g_best

    Returns the final position of the agent.
    """

    # ------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------
    current_route = beliefs.current_route
    current_cost = beliefs.current_cost

    if current_route is None:
        raise ValueError("[ERROR] Agent without initial solution defined")

    # ------------------------------------------------------------
    # 1 - Decision Method
    # ------------------------------------------------------------
    print("1 - Decision Method")
    action_name = decision_method(beliefs)
    print("Metaheuristic to be executed: ", action_name)
    
    if action_name not in METAHEURISTICS:
        raise ValueError(f"[ERROR] Metaheuristic '{action_name}' is not registered")
    action_fn = METAHEURISTICS[action_name]

    # ------------------------------------------------------------
    # 2 - Execution of the Metaheuristic (Action)
    # ------------------------------------------------------------
    print("2 - Execution of the Metaheuristic (Action)")
    new_route, new_cost = action_fn(current_route, instance)
    
    # Validate feasibility of the solution returned by the metaheuristic
    feasible, validated_cost = evaluate_route(new_route, instance)
    if not feasible:
        print(f"[WARNING] Metaheuristic '{action_name}' returned an infeasible solution. Keeping current solution.")
        new_route, new_cost = current_route, current_cost
    else:
        new_cost = validated_cost  # Use validated cost

    # ------------------------------------------------------------
    # 3 - Learning Method
    # ------------------------------------------------------------
    print("3 - Learning Method")
    beliefs.update_after_action(
        action_name=action_name,
        old_cost=current_cost,
        new_cost=new_cost
    )

    beliefs.update_current_solution(new_route, new_cost)

    # ------------------------------------------------------------
    # 4 - Velocity Operator (Path-Relinking)
    # ------------------------------------------------------------
    print("4 - Velocity Operator (Path-Relinking)")
    g_best_route, g_best_cost, _ = global_best.get()

    # If g_best does not exist, there is no diversification
    if g_best_route is not None:

        def opportunistic_intensification(route, instance):
            """
            Opportunistic intensification:
            choose the best metaheuristic according to the current beliefs.
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
        print("No g_best found, no diversification")
        final_route, final_cost = new_route, new_cost

    beliefs.update_current_solution(final_route, final_cost)

    # ------------------------------------------------------------
    # 5 - Update of p_best
    # ------------------------------------------------------------
    print("5 - Update of p_best")
    beliefs.try_update_pbest(final_route, final_cost)

    # ------------------------------------------------------------
    # 6 - Update of g_best (Blackboard)
    # ------------------------------------------------------------
    print("6 - Update of g_best (Blackboard)")
    global_best.try_update(
        candidate_route=final_route,
        candidate_cost=final_cost,
        agent_id=beliefs.agent_id
    )

    return final_route, final_cost
