import argparse
import sys

from sakhilabs.cli.inference import sakhi_inference_args
from sakhilabs.cli.train import sakhi_training_args


def main():
    """Main entry point for the Sakhi CLI"""
    parser = argparse.ArgumentParser(
        prog="sakhi", description="Sakhi CLI - Indic Language LLM development"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=False
    )

    sakhi_training_args(subparsers)
    sakhi_inference_args(subparsers)

    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
