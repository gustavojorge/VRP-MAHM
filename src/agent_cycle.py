from typing import Dict, Callable, Tuple, List

from src.agent_beliefs import AgentBeliefs
from src.shared import blackboard
from src.methods.decision_method import decision_method
from src.utils.logger import get_logger

from src.methods.path_relinking import path_relinking
from src.utils.evaluator import evaluate_route

# Available Metaheuristics
from src.actions.vnd.vnd import vnd
from src.actions.ils.ils import ils
from src.actions.vns.vns import vns

# Metaheuristics (Actions) registry
METAHEURISTICS: Dict[str, Callable[[List[int], dict], Tuple[List[int], float]]] = {
    "VND": vnd,
    "ILS": ils,
    "VNS": vns,
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
    logger = get_logger(beliefs.agent_id)
    action_name = decision_method(beliefs, logger=logger)
    logger.log_phase("[Decision Method] ")
    logger.log(f"---> Chosen Metaheuristic: {action_name}")
    
    if action_name not in METAHEURISTICS:
        raise ValueError(f"[ERROR] Metaheuristic '{action_name}' is not registered")
    action_fn = METAHEURISTICS[action_name]

    # ------------------------------------------------------------
    # 2 - Execution of the Metaheuristic (Action)
    # ------------------------------------------------------------
    logger.log_phase(f"[Action - Execution of {action_name} ]")
    new_route, new_cost = action_fn(current_route, instance)
    
    # Validate feasibility of the solution returned by the metaheuristic
    feasible, validated_cost = evaluate_route(new_route, instance)
    if not feasible:
        logger.log(f"[WARNING] Metaheuristic '{action_name}' returned an infeasible solution. Keeping current solution.")
        new_route, new_cost = current_route, current_cost
    else:
        new_cost = validated_cost  # Use validated cost

    # ------------------------------------------------------------
    # 3 - Learning Method
    # ------------------------------------------------------------
    logger.log_phase("[Learning Method - Updating beliefs after action]")
    beliefs.update_after_action(
        action_name=action_name,
        old_cost=current_cost,
        new_cost=new_cost
    )

    beliefs.update_current_solution(new_route, new_cost)

    # ------------------------------------------------------------
    # 4 - Velocity Operator (Path-Relinking)
    # ------------------------------------------------------------
    logger.log_phase("[Velocity Operator (Path-Relinking)]")
    # Access global_best from module to ensure we get the injected instance
    g_best_route, g_best_cost, _ = blackboard.global_best.get()

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
        logger.log("No g_best found, no diversification")
        final_route, final_cost = new_route, new_cost

    beliefs.update_current_solution(final_route, final_cost)

    # ------------------------------------------------------------
    # 5 - Update of p_best
    # ------------------------------------------------------------
    logger.log_phase("[Update of p_best]")
    old_p_best_cost = beliefs.p_best_cost
    p_best_updated = beliefs.try_update_pbest(final_route, final_cost)
    
    if p_best_updated:
        logger.log(f"---> p_best UPDATED: From {old_p_best_cost} to {beliefs.p_best_cost}")
    
    # ------------------------------------------------------------
    # 6 - Update of g_best (Blackboard)
    # ------------------------------------------------------------
    logger.log_phase("[Update of g_best (Blackboard)]")
    # Access global_best from module to ensure we get the injected instance
    g_route_before, g_cost_before, g_agent_before = blackboard.global_best.get()
    g_best_updated = blackboard.global_best.try_update(
        candidate_route=final_route,
        candidate_cost=final_cost,
        agent_id=beliefs.agent_id
    )
    
    if g_best_updated:
        logger.log(f"---> g_best UPDATED: From {g_cost_before if g_cost_before != float('inf') else 'inf'} to {final_cost} (by agent {beliefs.agent_id})")
    else:
        current_g_cost = g_cost_before if g_cost_before != float('inf') else 'inf'

    return final_route, final_cost
