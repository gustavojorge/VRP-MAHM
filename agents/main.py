from agents.agent_beliefs import AgentBeliefs
from agents.agent_cycle import run_agent_cycle
from agents.utils.generate_random_feasible_route import generate_random_feasible_route
from agents.problem.evaluator import evaluate_route

# Registro dos agentes ativos
AGENTS = {}

def initialize_agent(agent_id: str):
    instance = load_instance("instances/1.json")

    initial_route = generate_random_feasible_route(instance)
    feasible, cost = evaluate_route(initial_route, instance)

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
