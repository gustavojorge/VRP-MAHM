import multiprocessing as mp
from agents.shared.blackboard import GlobalBest
from agents.main import initialize_agent, run_cycle, AGENTS
from agents.utils.logger import get_logger, set_instance_name

def agent_worker(agent_id, max_iterations, global_blackboard, instance_path, instance_name):
    """
    Worker function for each agent process.
    
    Args:
        agent_id: Agent ID
        max_iterations: Maximum number of iterations
        global_blackboard: Shared GlobalBest instance
        instance_path: Path to the instance file
        instance_name: Name of the instance (for logging)
    """
    # Inject the shared blackboard into the module
    import agents.shared.blackboard
    agents.shared.blackboard.global_best = global_blackboard
    
    # Set instance name for logging
    set_instance_name(instance_name)
    
    # Initialize logger for this agent
    logger = get_logger(agent_id, instance_name)
    
    logger.log("===================================")
    logger.log(" STARTING AGENT EXECUTION ")
    logger.log("===================================")
    logger.log(f"Instance: {instance_name}")
    
    # Initialize the agent with the specified instance
    initialize_agent(agent_id, instance_path)
    
    beliefs = AGENTS[agent_id]["beliefs"]
    
    logger.log("\n--- Initial state ---")
    logger.log(f"Initial Solution/Route: {beliefs.current_route}")
    logger.log(f"Initial Cost: {beliefs.current_cost}")
    logger.log("----------------------\n")
    
    # Execute the agent cycle
    for iteration in range(max_iterations):
        logger.log(f"\n==== ITERATION {iteration} ====")
        
        run_cycle(agent_id)
        
        g_route, g_cost, g_agent = global_blackboard.get()
        logger.log_state(
            current_cost=beliefs.current_cost,
            p_best_cost=beliefs.p_best_cost,
            g_best_cost=g_cost if g_route is not None else None,
            g_best_agent=g_agent if g_route is not None else None
        )
    
    logger.log("\n===================================")
    logger.log(" EXECUTION FINALIZED ")
    logger.log("===================================")
    
    logger.log("\n--- Final result ---")
    logger.log(f"Agent's p_best cost: {beliefs.p_best_cost}")
    logger.log(f"Agent's p_best solution/route: {beliefs.p_best_route}")
    
    g_route, g_cost, g_agent = global_blackboard.get()
    logger.log(f"\nGlobal Best:")
    logger.log(f"Cost : {g_cost}")
    logger.log(f"Solution/Route  : {g_route}")

def run_for_instance(instance_path: str, instance_name: str, num_agents: int, max_iterations: int):
    """
    Run multi-agent system for a specific instance.
    
    Args:
        instance_path: Path to the instance file (e.g., "instances/1.json")
        instance_name: Name of the instance (for logging, e.g., "1" or "50")
        num_agents: Number of agents to run
        max_iterations: Maximum number of iterations per agent
    """
    mp.set_start_method("spawn", force=True)

    manager = mp.Manager()

    # Create the shared blackboard
    global_blackboard = GlobalBest(manager)

    processes = []

    # Create and start processes for each agent
    for i in range(num_agents):
        p = mp.Process(
            target=agent_worker,
            args=(f"agent_{i}", max_iterations, global_blackboard, instance_path, instance_name)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Display final result
    g_route, g_cost, g_agent = global_blackboard.get()
    if g_route is not None:
        print(f"\n=== Instance {instance_name} - Global Best ===")
        print(f"Cost: {g_cost}")
        print(f"Agent: {g_agent}")
        print(f"Route: {g_route}")
    else:
        print(f"\n=== Instance {instance_name} - No global best found ===")

def main():
    NUM_AGENTS = 2
    MAX_ITERATIONS = 10
    
    # Default: run for instance 50.json
    run_for_instance("instances/50.json", "50", NUM_AGENTS, MAX_ITERATIONS)

if __name__ == "__main__":
    main()
