import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Read metrics from a workflow run.")
    parser.add_argument("--run-dir", required=True, help="Path to outputs/runs/<run_id>.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = Path(args.run_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
