import flwr as fl
from argparse import ArgumentParser
from client import *
from typing import Callable, Dict, List, Optional, Tuple


def get_eval_fn() -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    # pylint: disable=no-member

    # pylint: enable=no-member

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        set_parameters(weights)
        return model_test(model, tensor_val_mask)

    return evaluate


def set_parameters(parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def fit_config(rnd: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config: Dict[str, fl.common.Scalar] = {
        "epoch_global": str(rnd),
        "epochs": str(5),
        "batch_size": str(128),
    }
    return config


if __name__ == "__main__":
    parser = ArgumentParser(description="GCN Server")
    parser.add_argument(
        "--server_address",
        type=str,
        default="[::]:8080",
        help=f"gRPC server address (default: '[::]:8080')",
    )
    parser.add_argument(
        "--fraction_fit",
        type=float,
        default=0.2,
        help=f"Fraction of clients used during training. Defaults to 0.1.",
    )
    parser.add_argument(
        "--fraction_eval",
        type=float,
        default=0.1,
        help=f"Fraction of clients used during validation. Defaults to 0.1.",
    )
    parser.add_argument(
        "--min_fit_clients",
        type=int,
        default=5,
        help=f"Minimum number of clients used during training. Defaults to 2.",
    )
    parser.add_argument(
        "--min_eval_clients",
        type=int,
        default=5,
        help=f"Minimum number of clients used during validation. Defaults to 2.",
    )

    parser.add_argument(
        "--min_available_clients",
        type=int,
        default=5,
        help=f"Minimum number of total clients in the system.Defaults to 5.",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help=f"Total training round. Default to 3.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=f"Adaptive alpha. Default to 0.5.",
    )

    args = parser.parse_args()

    strategy = fl.server.strategy.FedAsync(
        fraction_fit=args.fraction_fit,
        fraction_eval=args.fraction_eval,
        min_fit_clients=args.min_fit_clients,
        min_eval_clients=args.min_eval_clients,
        min_available_clients=args.min_available_clients,
        eval_fn=get_eval_fn(),
    )
    client_manager = fl.server.client_manager.SimpleClientManager()
    server = fl.server.AsyncServer(client_manager, strategy, alpha=args.alpha)

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": args.rounds},
        server=server,
    )
