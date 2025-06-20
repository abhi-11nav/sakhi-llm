import yaml

from sakhi.configs.utils.config import SakhiConfig
from sakhi.pipelines.train.pretraining import main


def sakhi_training_args(subparsers):
    """Add training arguments to the subparser"""
    train_parser = subparsers.add_parser(
        "train",
        help="Train the model",
        description="Train a machine learning model with specified parameters",
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
    main(config=sakhi_config)
