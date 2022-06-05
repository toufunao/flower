import flwr as fl
from argparse import ArgumentParser

if __name__ == '__main__':
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
        default=2,
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
        default=3,
        help=f"Total training round. Default to 3.",
    )

    args = parser.parse_args()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_eval=args.fraction_eval,
        min_fit_clients=args.min_fit_clients,
        min_eval_clients=args.min_eval_clients,
        min_available_clients=args.min_available_clients
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": args.rounds},
        strategy=strategy,
    )
