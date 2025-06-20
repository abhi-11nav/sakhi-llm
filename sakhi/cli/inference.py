import yaml

from sakhi.configs.utils.config import SakhiConfig
from sakhi.pipelines.inference.inference import main


def sakhi_inference_args(subparsers):
    """Add training arguments to the subparser"""
    inference_parser = subparsers.add_parser(
        "inference",
        help="Run model inference",
        description="Sakhi inference",
    )

    inference_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config for model and inference parameters",
    )

    inference_parser.add_argument(
        "--prompt",
        type=str,
        default="చెప్పు",
        help="Prompt for model inference",
    )

    inference_parser.set_defaults(func=sakhi_inference)


def sakhi_inference(args):
    config = SakhiConfig._load_config(config_path=args.config)
    main(config=config, prompt=args.prompt)
