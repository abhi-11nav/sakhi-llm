import argparse
import sys

from sakhilabs.cli.model.inference_model import sakhi_inference_args
from sakhilabs.cli.model.train_model import sakhi_training_args
from sakhilabs.cli.tokenizer.test_tokenizer import sakhi_tokenizer_test_args
from sakhilabs.cli.tokenizer.train_tokenizer import sakhi_tokenizer_train_args


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
    sakhi_tokenizer_train_args(subparsers)
    sakhi_tokenizer_test_args(subparsers)

    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
