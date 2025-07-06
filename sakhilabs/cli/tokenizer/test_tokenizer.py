from sakhilabs.pipelines.tokenizer.test_tokenizer import analyze_tokenizer


def sakhi_tokenizer_test_args(subparsers):
    test_tokenizer = subparsers.ArgumentParser(
        description="Analyze tokenizer performance on a JSONL file"
    )
    test_tokenizer.add_argument(
        "--tokenizer", required=True, help="Path to tokenizer directory"
    )
    test_tokenizer.add_argument("--data_path", required=True, help="Path to JSONL file")
    test_tokenizer.add_argument(
        "--max_lines", type=int, default=100000, help="Number of lines to sample"
    )

    test_tokenizer.set_defaults(func=do_test_tokenizer)


def do_test_tokenizer(args):
    analyze_tokenizer(args.tokenizer, args.jsonl, args.max_lines)
