import yaml

from sakhilabs.configs.utils.config import SakhiConfig
from sakhilabs.pipelines.train import instruction_tuning_run, pretraining_run


def sakhi_training_args(subparsers):
    """Add training arguments to the subparser"""
    train_parser = subparsers.add_parser(
        "train",
        help="Train the model",
        description="Train a machine learning model with specified parameters",
    )
    train_parser.add_argument(
        "--type",
        type=str,
        choices=["pretrain", "instruction_tune"],
        required=True,
        help="Type of training: 'pretrain' or 'instruction_tune'",
    )

    train_parser.add_argument(
        "--config",
        type=str,
        default="sakhi/configs/sakhi_telugu__681M.yaml",
        help="Config for model parameters and training",
    )

    train_parser.set_defaults(func=do_train)


def do_train(args):
    sakhi_config = SakhiConfig._load_config(config_path=args.config)

    if args.type == "pretrain":
        pretraining_run(config=sakhi_config)
    elif args.type == "instruction_tune":
        instruction_tuning_run(config=sakhi_config)
    else:
        raise ValueError("Invalid Options !!")
