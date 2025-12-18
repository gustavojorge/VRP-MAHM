import multiprocessing as mp
from agents.shared.blackboard import GlobalBest
from agents.main import initialize_agent, run_cycle, AGENTS
from agents.utils.logger import get_logger

def agent_worker(agent_id, max_iterations, global_blackboard):
    """
    Worker function for each agent process.
    
    Args:
        agent_id: Agent ID
        max_iterations: Maximum number of iterations
        global_blackboard: Shared GlobalBest instance
    """
    # Inject the shared blackboard into the module
    import agents.shared.blackboard
    agents.shared.blackboard.global_best = global_blackboard
    
    # Initialize logger for this agent
    logger = get_logger(agent_id)
    
    logger.log("===================================")
    logger.log(" STARTING AGENT EXECUTION ")
    logger.log("===================================")
    
    # Initialize the agent
    initialize_agent(agent_id)
    
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

def main():
    NUM_AGENTS = 2
    MAX_ITERATIONS = 10

    mp.set_start_method("spawn", force=True)

    manager = mp.Manager()

    # Create the shared blackboard
    global_blackboard = GlobalBest(manager)

    processes = []

    # Create and start processes for each agent
    for i in range(NUM_AGENTS):
        p = mp.Process(
            target=agent_worker,
            args=(f"agent_{i}", MAX_ITERATIONS, global_blackboard)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Display final result
    g_route, g_cost, g_agent = global_blackboard.get()
    if g_route is not None:
        print(f"\n=== Global Best ===")
        print(f"Cost: {g_cost}")
        print(f"Agent: {g_agent}")
        print(f"Route: {g_route}")

if __name__ == "__main__":
    main()
