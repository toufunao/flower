import flwr as fl

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAsync(
        fraction_fit=0.5,
        fraction_eval=0.5,
    )
    client_manager = fl.server.client_manager.SimpleClientManager()
    server = fl.server.AsyncServer(client_manager, strategy)
    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 3},
        server=server,
    )
