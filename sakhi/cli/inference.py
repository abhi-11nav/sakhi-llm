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
        default="<|instruction|> సాఫ్ట్‌వేర్ ఇన్స్టాల్ చేయడానికి సులభమైన దశలను చెప్పండి. <|response|> ",
        help="Telugu prompt for the model to reply",
    )

# <|instruction|> ఈ ప్రయోగంలో మీరు ఏ పద్ధతిని అనుసరించారో వివరించండి. విద్యుత్ ప్రవాహం కొలవడానికి మేము అమిత్ గేజ్ ఉపయోగించాము. <|response|>
# <|instruction|> ఈ పద్యం యొక్క భావాన్ని వివరించండి. తల్లి ప్రేమను వర్ణించే ఈ పద్యంలో, ఆమె త్యాగాలను కవి వివరించారు. <|response|>
# <|instruction|> మీరు ఈ సమస్యను ఎలా పరిష్కరించారో వివరించండి. సమీకరణాలను సరియైన పద్ధతిలో పరిష్కరించడానికి మేము తొలి సూత్రాన్ని ఉపయోగించాము. <|response|>

    inference_parser.set_defaults(func=sakhi_inference)


def sakhi_inference(args):
    config = SakhiConfig._load_config(config_path=args.config)
    main(config=config, prompt=args.prompt)
