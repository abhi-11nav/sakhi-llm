import yaml

from sakhilabs.model.model import SakhiModel


def get_model_param():
    with open("sakhi/configs/sakhi_telugu__764M.yaml", "r") as file:
        config = yaml.safe_load(file)

    model = SakhiModel(
        embed_dim=config["model_parameters"]["embed_dim"],
        num_heads=config["model_parameters"]["num_heads"],
        ff_dim=config["model_parameters"]["ff_dim"],
        vocab_size=config["model_parameters"]["vocab_size"],
        num_layers=config["model_parameters"]["num_layers"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params} parameters")


if __name__ == "__main__":
    get_model_param()
