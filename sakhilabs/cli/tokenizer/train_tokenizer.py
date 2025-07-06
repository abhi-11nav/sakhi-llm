from sakhilabs.pipelines.tokenizer.train_tokenzier import train_tokenizer


def sakhi_tokenizer_train_args(subparsers):
    tokenizer_train_parser = subparsers.ArgumentParser(
        description="Train a BPE tokenizer on JSONL corpus"
    )
    tokenizer_train_parser.add_argument(
        "--corpus", type=str, required=True, help="Path to input .jsonl file"
    )
    tokenizer_train_parser.add_argument(
        "--save_dir", type=str, default="sakhi_tokenizer", help="Save directory"
    )
    tokenizer_train_parser.add_argument(
        "--vocab_size", type=int, help="Vocabulary size"
    )
    tokenizer_train_parser.add_argument(
        "--min_frequency", type=int, default=2, help="Minimum frequency"
    )

    tokenizer_train_parser.set_defaults(func=do_train_tokenizer)


def do_train_tokenizer(args):
    train_tokenizer(
        corpus_file=args.corpus,
        save_dir=args.save_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
