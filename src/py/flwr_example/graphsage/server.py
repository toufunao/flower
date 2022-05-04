import flwr

if __name__ == '__main__':
    flwr.server.start_server(config={"num_rounds": 3})
