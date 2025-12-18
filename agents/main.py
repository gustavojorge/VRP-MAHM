from agents.agent_beliefs import AgentBeliefs
from agents.agent_cycle import run_agent_cycle
from agents.utils.generate_random_feasible_route import generate_random_feasible_route
from agents.problem.evaluator import evaluate_route
from agents.utils.load_instance import load_instance

# Active agents registry
AGENTS = {}

def initialize_agent(agent_id: str):
    instance = load_instance("instances/1.json")

    # Generate initial feasible route with validation
    max_attempts = 100
    initial_route = None
    feasible = False
    cost = float("inf")
    
    for attempt in range(max_attempts):
        try:
            candidate_route = generate_random_feasible_route(instance)
            feasible, candidate_cost = evaluate_route(candidate_route, instance)
            
            if feasible:
                initial_route = candidate_route
                cost = candidate_cost
                break
        except RuntimeError:
            # If generation fails, try again
            continue
    
    if not feasible or initial_route is None:
        raise RuntimeError(
            f"[ERROR] It was not possible to generate an initial feasible route after {max_attempts} attempts"
        )

    beliefs = AgentBeliefs(
        agent_id=agent_id,
        metaheuristics=["VND", "ILS"]
    )

    beliefs.update_current_solution(initial_route, cost)
    beliefs.try_update_pbest(initial_route, cost)

    AGENTS[agent_id] = {
        "beliefs": beliefs,
        "instance": instance
    }


def run_cycle(agent_id: str):
    data = AGENTS[agent_id]
    beliefs = data["beliefs"]
    instance = data["instance"]

    run_agent_cycle(beliefs, instance)
