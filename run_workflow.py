import argparse

from src.workflow.graph import run_workflow


def parse_args():
    parser = argparse.ArgumentParser(description="Run the agentic NER evaluation workflow.")
    parser.add_argument("--config", required=True, help="Path to a YAML workflow config.")
    return parser.parse_args()


def main():
    args = parse_args()
    final_state = run_workflow(args.config)
    print("Workflow completed.")
    print(f"Run directory: {final_state['run_dir']}")
    print(f"Report: {final_state['report_path']}")


if __name__ == "__main__":
    main()
