from modules.agent import AgentOrchestrator
import time
import os
import sys
def main(task: str = None):
    """Main function to execute the AgentOrchestrator."""
    # Create an instance of the AgentOrchestrator
    log_path = os.path.join("log_results", time.strftime("%Y-%m-%d_%H-%M-%S"))
    orchestrator = AgentOrchestrator(save_path=log_path, save_result=True)
    if task is None:
        # Default task if none is provided
        task = "Find nearby McDonald's, plan the route and choose public transport."
    # Run the orchestrator
    orchestrator.run(task)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)