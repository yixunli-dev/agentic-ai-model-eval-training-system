import argparse

from src.workflow.graph import run_workflow


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM NER through the workflow.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    final_state = run_workflow(args.config)
    print(f"Training workflow finished: {final_state['run_dir']}")


if __name__ == "__main__":
    main()
