from modules.agent import AgentOrchestrator

def main() -> None:
    """Main function to execute the AgentOrchestrator."""
    # Create an instance of the AgentOrchestrator
    orchestrator = AgentOrchestrator()
    task: str = "Open Amazon and search for 'laptop', filter by 'free shipping'."
    # Run the orchestrator
    orchestrator.run(task)

if __name__ == "__main__":
    main()